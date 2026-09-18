"""
desensitize.py — PII removal from cleaned patient queries.

Each cleaned query is passed to a locally served Ollama model that is asked to
replace only four categories of directly identifying information (personal name,
phone number, national ID, medical-record number) and to return the list of
categories it acted on. The prompt is deliberately conservative: when the model
is not certain that a span identifies a person, it must leave the text unchanged.

Pipeline position
    raw transcripts -> clean.py -> desensitize.py -> dedup.py -> annotation/

Usage
    python preprocessing/desensitize.py \
        --input  data/processed/cleaned.csv \
        --output data/processed/desensitized.csv \
        --model  llama3.1:latest \
        --texts-only data/processed/desensitized_texts_only.csv

Input
    CSV with a text column; one of cleaned_text / query / text is picked up
    automatically. The input column is copied to output as original_text.

Output columns
    original_text, desensitized_text, sensitive_types, reason

`sensitive_types` is a JSON-array string, e.g. '["姓名"]' or '[]'. Rows carrying
a non-empty list are the subset in which a replacement was made. The optional
--texts-only path receives a single-column frame for pipelines that only need
the de-identified text.

The run is resumable via a checkpoint written next to the output file.
"""

import argparse
import itertools
import json
import os
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock

import pandas as pd
import requests

warnings.filterwarnings("ignore")

DEFAULT_ENDPOINTS = [
    "http://127.0.0.1:11434/api/generate",
    "http://127.0.0.1:11435/api/generate",
    "http://127.0.0.1:11436/api/generate",
    "http://127.0.0.1:11437/api/generate",
]
TEXT_COLUMNS = ["cleaned_text", "query", "text", "desensitized_text"]
OUTPUT_COLUMNS = ["original_text", "desensitized_text", "sensitive_types", "reason"]

SYSTEM_PROMPT = """你是一个严谨的医疗数据脱敏专家。你的任务是根据上下文，精准识别并替换**明确指向具体个人身份**的信息。

【关键原则】
- 不要臆测！不要凭单个词语或数字就做判断。
- 只有当你**100%确定**它是一个真实的人名、电话、ID或病历号时，才进行替换。
- **当你不确定时，绝对不要做任何改动，保留原文本。**

【只脱敏以下4类明确信息】
1.  **姓名 (Name)**:
    - 规则：完整的"姓+名"组合（如张三、李四医生、王主任）。
    - 示例："张伟" → "[姓名]"；"刘医生" → "[姓名]医生"。
    - **绝不脱敏**：单独的"张"、"主任"、"周日"、"一号"。

2.  **电话号码 (Phone)**:
    - 规则：11位手机号 或 带区号的座机号（如 010-12345678）。
    - **绝不脱敏**：任何不完整的数字串、或没有"电话"前缀引导的数字（如"510分钟"、"420号"）。

3.  **身份证号 (ID)**:
    - 规则：18位（含X）或15位纯数字串。
    - **绝不脱敏**：其他任何数字串。

4.  **病历号/住院号 (Medical Record)**:
    - 规则：由**"病历号"、"住院号"、"登记号"**等明确前缀引导的字母/数字组合。
    - 示例："住院号20230615" → "住院号[病历号]"；"病历号：A1234" → "病历号：[病历号]"。
    - **绝不脱敏**：没有明确前缀引导的纯数字，如"420"、"1号"、"3次"。

【输出格式】
只输出JSON：
{"desensitized_text": "脱敏后的文本", "sensitive_types": ["姓名"]}
如果没有敏感信息，返回 {"desensitized_text": "原文本", "sensitive_types": []}"""


def build_prompt(text):
    return f"""脱敏以下文本，只替换姓名、电话、身份证、病历号这4类明确信息：
原文：{text}

只输出JSON。"""


def _strip_code_fence(content):
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _parse_json(content, original, fallback_reason):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", content)
        if not match:
            return {"desensitized_text": original, "sensitive_types": [], "reason": fallback_reason}
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return {"desensitized_text": original, "sensitive_types": [], "reason": fallback_reason}
    if not isinstance(parsed, dict):
        return {"desensitized_text": original, "sensitive_types": [], "reason": "unexpected JSON shape"}
    return parsed


def call_llm(text, endpoint_cycle, model, max_retries=2):
    url = next(endpoint_cycle)
    full_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{build_prompt(text)}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>"
    )
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512, "top_p": 0.9, "stop": ["<|eot_id|>"]},
    }
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            content = _strip_code_fence(response.json().get("response", "").strip())
            if not content:
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return {"desensitized_text": text, "sensitive_types": [], "reason": "empty model response"}
            return _parse_json(content, text, "JSON parse failed")
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"desensitized_text": text, "sensitive_types": [], "reason": f"request error: {str(exc)[:40]}"}
    return {"desensitized_text": text, "sensitive_types": [], "reason": "retries exhausted"}


def process_batch(batch, indices, store, lock, endpoint_cycle, model):
    for text, idx in zip(batch, indices):
        result = call_llm(text, endpoint_cycle, model)
        types = result.get("sensitive_types") or []
        if not isinstance(types, list):
            types = [str(types)]
        with lock:
            store[idx] = {
                "original_text": text,
                "desensitized_text": result.get("desensitized_text", text),
                "sensitive_types": json.dumps(types, ensure_ascii=False),
                "reason": result.get("reason", ""),
            }


def _write(output_path, texts_only_path, rows):
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    if texts_only_path:
        frame[["desensitized_text"]].to_csv(texts_only_path, index=False, encoding="utf-8-sig")


def _save_checkpoint(path, done):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"processed_indices": sorted(done), "timestamp": datetime.now().isoformat()},
                  handle, ensure_ascii=False, indent=2)


def _load_checkpoint(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return set(json.load(handle).get("processed_indices", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def main():
    parser = argparse.ArgumentParser(description="Remove directly identifying information")
    parser.add_argument("--input", required=True, help="cleaned corpus CSV")
    parser.add_argument("--output", required=True, help="de-identified corpus CSV")
    parser.add_argument("--texts-only", default=None, help="optional single-column output path")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--endpoints", nargs="*", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sample", type=int, default=None, help="process only the first N rows")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "desensitize_checkpoint.json")

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    if args.sample:
        frame = frame.head(args.sample)
    text_col = next((c for c in TEXT_COLUMNS if c in frame.columns), None)
    if text_col is None:
        raise SystemExit(f"no text column found; columns are {frame.columns.tolist()}")

    texts = frame[text_col].astype(str).tolist()
    total = len(texts)
    done = _load_checkpoint(checkpoint_path)
    store = {}
    if os.path.exists(args.output):
        previous = pd.read_csv(args.output, encoding="utf-8-sig")
        for position, row in enumerate(previous.itertuples(index=False)):
            store[position] = dict(zip(OUTPUT_COLUMNS, list(row)[: len(OUTPUT_COLUMNS)]))

    pending = [(i, texts[i]) for i in range(total) if i not in done]
    print(f"[INFO] rows {total} | resumed {len(done)} | pending {len(pending)}")
    if not pending:
        _write(args.output, args.texts_only, [store[i] for i in sorted(store)])
        print("[OK] nothing to do")
        return

    lock = Lock()
    cycle = itertools.cycle(args.endpoints)
    batches = [pending[i:i + args.batch] for i in range(0, len(pending), args.batch)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for number, chunk in enumerate(batches, 1):
            future = executor.submit(
                process_batch, [c[1] for c in chunk], [c[0] for c in chunk],
                store, lock, cycle, args.model,
            )
            future.result()
            done.update(index for index, _ in chunk)
            _save_checkpoint(checkpoint_path, done)
            _write(args.output, None, [store[i] for i in sorted(store)])
            elapsed = max(time.time() - started, 1e-6)
            print(f"[INFO] {len(done)}/{total} ({len(done) / total * 100:.1f}%) "
                  f"| batch {number}/{len(batches)} | {len(done) / elapsed * 60:.1f} rows/min")

    rows = [store[i] for i in sorted(store)]
    _write(args.output, args.texts_only, rows)
    final = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    flagged = int((final["sensitive_types"] != "[]").sum())
    counter = Counter()
    for value in final["sensitive_types"]:
        try:
            counter.update(json.loads(value))
        except json.JSONDecodeError:
            continue
    print(f"[OK] {len(final)} rows | replaced in {flagged} | types {dict(counter.most_common())}")
    print(f"[OK] written to {args.output}")


if __name__ == "__main__":
    main()
