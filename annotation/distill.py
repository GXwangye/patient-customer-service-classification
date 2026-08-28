"""
distill.py — 标签蒸馏流水线（候选标签 → 一级/二级标签体系）

将多 LLM 归纳出的细粒度候选业务描述，经六节点 LLM 蒸馏，收敛为规范化的
一级 / 二级标签体系，并产出"原始描述 → 一级 / 二级标签"映射。

输入：含 `business_description` 列的 CSV（候选标签归纳阶段产出）。
输出：tagged_descriptions.csv、node*.json、FULL_REPORT.json（默认写入 data/processed/distillation）。

依赖：pip install -r requirements.txt
运行：python distill.py --input data/processed/distill_input/pre_annotated_data.csv
"""

import json
import os
import argparse
import itertools
import time
import random
from collections import Counter
from datetime import datetime

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import warnings
warnings.filterwarnings("ignore")


# ===================== 默认配置 =====================
DEFAULT_ENDPOINTS = [
    "http://127.0.0.1:11434/api/generate",
    "http://127.0.0.1:11435/api/generate",
    "http://127.0.0.1:11436/api/generate",
    "http://127.0.0.1:11437/api/generate",
]
DEFAULT_INPUT = "data/processed/distill_input/pre_annotated_data.csv"
DEFAULT_OUTPUT = "data/processed/distillation"
BATCH_SIZE = 30
MAX_WORKERS = 8
TEST_MODE = False

lock = Lock()
stats = {"total": 0, "success": 0, "fail": 0}


def call_llm(prompt, model_name, endpoint_cycle, max_retries=3):
    """通用 LLM 调用（纯文本 JSON 输出，线程安全）。"""
    with lock:
        url = next(endpoint_cycle)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 4096, "top_p": 0.9},
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            content = r.json().get("response", "").strip()
            if not content:
                time.sleep(2)
                continue
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            m = re.search(r"\[.*\]", content, re.DOTALL) if "[" in content else re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                content = m.group()
            parsed = json.loads(content)
            return parsed if isinstance(parsed, (dict, list)) else None
        except (json.JSONDecodeError, Exception):  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


import re  # noqa: E402  (used above; kept central for clarity)


def save_checkpoint(data, filename, out_dir):
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_sample(data, label, sample_size=5):
    print(f"\n[抽查] {label}（随机{sample_size}条）:")
    if not data:
        print("  无数据")
        return
    for i, item in enumerate(random.sample(data, min(sample_size, len(data))), 1):
        print(f"   {i}. {json.dumps(item, ensure_ascii=False)[:200]}")


DISTILL_PROMPT = """你是医院业务分类专家。为每条业务描述生成一级标签和二级标签。

【一级标签规则】从业务板块角度归类（如：门诊服务、住院服务、医保服务、药品服务、医院流程、投诉建议）
【二级标签规则】在一级标签下的具体业务环节（如：挂号咨询、检查报告、费用结算）
【注意】每条描述独立判断，不要受其他描述影响。

输入描述：
{descriptions}

输出JSON数组：[{"description": "原始描述", "level1": "一级标签", "level2": "二级标签"}]"""


MERGE_LEVEL1_PROMPT = """你是医院业务分类专家。以下是一组一级标签，请合并语义相近的标签。
【当前一级标签】
{tags}
【要求】1. 合并语义相近的标签；2. 保留必要的区分度；3. 输出精简后的标签列表。
【输出格式】只返回JSON数组：["标签1", "标签2", ...]"""

MERGE_LEVEL2_PROMPT = """你是医院业务分类专家。以下是一组二级标签，请合并语义相近的标签。
【当前二级标签】
{tags}
【要求】1. 合并语义相近的标签；2. 保留必要的区分度；3. 输出精简后的标签列表。
【输出格式】只返回JSON数组：["标签1", "标签2", ...]"""


def main():
    ap = argparse.ArgumentParser(description="标签蒸馏流水线")
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--endpoints", nargs="*", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    endpoint_cycle = itertools.cycle(args.endpoints)

    # 节点1：提取唯一业务描述
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    descs = df["business_description"].dropna().tolist()
    uniq = list({d.strip() for d in descs if d and d.strip()})
    print(f"[节点1] 原始 {len(df)} 条 → 唯一描述 {len(uniq)} 条")
    save_checkpoint({"唯一业务描述数": len(uniq)}, "node1_unique_descriptions_stats.json", args.output)
    save_checkpoint(uniq, "node1_unique_descriptions_full.json", args.output)

    # 节点2：分批蒸馏（线程池并发）
    batches = [uniq[i : i + args.batch] for i in range(0, len(uniq), args.batch)]
    all_tagged = []
    print(f"[节点2] 共 {len(batches)} 批，并发 {args.workers}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for bi, batch in enumerate(batches):
            futs[ex.submit(_process_batch, batch, bi, args, endpoint_cycle)] = bi
        for fut in as_completed(futs):
            bi, valid_items = fut.result()
            all_tagged.extend(valid_items)
            print(f"   批次 {bi+1}/{len(batches)} 完成，成功 {len(valid_items)} 条")
    pd.DataFrame(all_tagged).to_csv(os.path.join(args.output, "node2_all_batch_results.csv"), index=False, encoding="utf-8-sig")

    # 节点3/4：一、二级标签合并
    l1 = list({it.get("一级标签", "未分类") for it in all_tagged})
    l2 = list({it.get("二级标签", "未分类") for it in all_tagged})
    merged_l1 = call_llm(MERGE_LEVEL1_PROMPT.replace("{tags}", "\n".join(f"- {t}" for t in l1)), args.model, endpoint_cycle) or l1
    merged_l2 = call_llm(MERGE_LEVEL2_PROMPT.replace("{tags}", "\n".join(f"- {t}" for t in l2)), args.model, endpoint_cycle) or l2
    save_checkpoint({"合并前": len(l1), "合并后": len(merged_l1)}, "node3_level1_merge_stats.json", args.output)
    save_checkpoint({"合并前": len(l2), "合并后": len(merged_l2)}, "node4_level2_merge_stats.json", args.output)

    # 节点5：标签映射与规范化
    l1_map = {o: next((n for n in merged_l1 if o == n or o in n or n in o), o) for o in l1}
    l2_map = {o: next((n for n in merged_l2 if o == n or o in n or n in o), o) for o in l2}
    final_tagged = [
        {
            "description": it.get("原始描述", ""),
            "level1": l1_map.get(it.get("一级标签", "未分类"), it.get("一级标签")),
            "level2": l2_map.get(it.get("二级标签", "未分类"), it.get("二级标签")),
        }
        for it in all_tagged
    ]
    save_checkpoint({"一级映射": l1_map, "二级映射": l2_map}, "node5_tag_mapping.json", args.output)

    # 节点6：最终统计
    l1_cnt = Counter(t["level1"] for t in final_tagged)
    l2_cnt = Counter(t["level2"] for t in final_tagged)
    report = {
        "处理时间": datetime.now().isoformat(),
        "配置": {"模型": args.model, "批次": args.batch, "并发": args.workers},
        "标签精简效果": {
            "一级": f"{len(l1)} -> {len(merged_l1)}",
            "二级": f"{len(l2)} -> {len(merged_l2)}",
        },
        "一级标签分布": dict(l1_cnt.most_common()),
    }
    save_checkpoint(report, "FULL_REPORT.json", args.output)
    pd.DataFrame(final_tagged).to_csv(os.path.join(args.output, "tagged_descriptions.csv"), index=False, encoding="utf-8-sig")
    print(f"[完成] 唯一描述 {len(uniq)} → 最终标签 {len(final_tagged)} 条")
    print(f"[完成] 一级标签 {len(l1)} → {len(merged_l1)}；二级标签 {len(l2)} → {len(merged_l2)}")


def _process_batch(batch, batch_idx, args, endpoint_cycle):
    desc_list = "\n".join(f"{i+1}. {d}" for i, d in enumerate(batch))
    prompt = DISTILL_PROMPT.replace("{descriptions}", desc_list)
    result = call_llm(prompt, args.model, endpoint_cycle)
    items = []
    if isinstance(result, list):
        for it in result:
            if isinstance(it, dict) and "description" in it:
                items.append({
                    "批次": batch_idx + 1,
                    "原始描述": it.get("description", ""),
                    "一级标签": it.get("level1", ""),
                    "二级标签": it.get("level2", ""),
                })
    return batch_idx, items


if __name__ == "__main__":
    main()
