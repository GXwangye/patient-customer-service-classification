"""
clean.py — patient-query extraction from raw customer-service transcripts.

Raw transcripts interleave patient questions with agent replies and small talk.
A locally served Ollama model extracts the patient-side query and flags whether
the text carries a usable question. The output of this step is the cleaned
corpus consumed by desensitize.py.

Pipeline position
    raw transcripts -> clean.py -> desensitize.py -> dedup.py -> annotation/

Usage
    python preprocessing/clean.py \
        --input data/raw/raw_transcripts.csv \
        --output data/processed/cleaned.csv \
        --model llama3.1:latest

Input
    CSV with a text column; one of query / text / content / question is picked up
    automatically.

Output columns
    original_text, cleaned_text, is_valid, is_complete, reason

The run is resumable: progress is checkpointed after every batch, and re-running
with the same --output skips texts already present in the output file.
"""

import argparse
import itertools
import json
import os
import re
import time
import warnings
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
TEXT_COLUMNS = ["query", "text", "content", "question", "cleaned_text"]
OUTPUT_COLUMNS = ["original_text", "cleaned_text", "is_valid", "is_complete", "reason"]

SYSTEM_PROMPT = """你是医疗文本清洗专家。从客服对话中提取患者咨询，剔除噪音。

规则：
1. 只保留患者提问，删除客服回复
2. 剔除"嗯、好、收到"等无意义词
3. 多个问题全部保留
4. 保留包含症状或疑问的文本

输出JSON格式：{"cleaned_text":"清洗后文本","is_valid":true/false,"is_complete":true/false,"reason":"原因"}"""

FAILED = {"cleaned_text": "", "is_valid": False, "is_complete": False, "reason": ""}


def build_prompt(text):
    return f"""清洗以下文本，只输出JSON：

原始：{text}

输出JSON："""


def _strip_code_fence(content):
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _parse_json(content, fallback_reason):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", content)
        if not match:
            return dict(FAILED, reason=fallback_reason)
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return dict(FAILED, reason=fallback_reason)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    return dict(FAILED, reason="unexpected JSON shape")


def call_llm(text, endpoint_cycle, model, max_retries=2):
    url = next(endpoint_cycle)
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\n请清洗以下文本，只输出JSON格式，不要其他文字：\n\n原始文本：{text}\n\n输出JSON：",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256, "top_p": 0.9},
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
                return dict(FAILED, reason="empty model response")
            return _parse_json(content, "JSON parse failed")
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return dict(FAILED, reason=f"request error: {str(exc)[:40]}")
    return dict(FAILED, reason="retries exhausted")


def process_batch(batch, indices, store, lock, endpoint_cycle, model):
    for text, idx in zip(batch, indices):
        result = call_llm(text, endpoint_cycle, model)
        with lock:
            store[idx] = {
                "original_text": text,
                "cleaned_text": result.get("cleaned_text", ""),
                "is_valid": bool(result.get("is_valid", False)),
                "is_complete": bool(result.get("is_complete", False)),
                "reason": result.get("reason", ""),
            }


def _write(output_path, rows):
    if rows:
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )


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
    parser = argparse.ArgumentParser(description="Extract patient queries from raw transcripts")
    parser.add_argument("--input", required=True, help="raw transcripts CSV")
    parser.add_argument("--output", required=True, help="cleaned corpus CSV")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--endpoints", nargs="*", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample", type=int, default=None, help="process only the first N rows")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "clean_checkpoint.json")

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
            _write(args.output, [store[i] for i in sorted(store)])
            elapsed = max(time.time() - started, 1e-6)
            print(f"[INFO] {len(done)}/{total} ({len(done) / total * 100:.1f}%) "
                  f"| batch {number}/{len(batches)} | {len(done) / elapsed * 60:.1f} rows/min")

    _write(args.output, [store[i] for i in sorted(store)])
    final = pd.DataFrame([store[i] for i in sorted(store)], columns=OUTPUT_COLUMNS)
    valid = int(final["is_valid"].sum())
    print(f"[OK] {valid}/{len(final)} valid queries ({valid / len(final) * 100:.1f}%) -> {args.output}")


if __name__ == "__main__":
    main()
