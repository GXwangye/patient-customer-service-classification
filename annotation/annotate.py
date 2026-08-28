"""
annotate.py — 多大型语言模型协同标注（单模型标注脚本）

对脱敏后的患者客服咨询文本，调用本地 Ollama 部署的大模型（GLM / Qwen / LLaMA / DeepSeek）
输出 20 个一级业务标签中的唯一标签。三个模型各自独立运行本脚本（仅改 MODEL_NAME），
其输出经投票、分歧识别与复核后形成训练标签（见论文 §2.3–§2.4）。

依赖：pip install -r requirements.txt
运行：python annotate.py --model qwen3:latest --input data/raw/desensitized_texts_only.csv
说明：
  - 默认对 4 个本地 Ollama 端口（11434–11437）做轮询并发；可按显存改为单端口。
  - 支持断点续跑（checkpoint）与批次保存。
  - 输出 CSV 列：original_text, label, reason。
"""

import json
import time
import os
import argparse
import itertools
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ===================== 默认配置（可用命令行参数覆盖） =====================
DEFAULT_ENDPOINTS = [
    "http://127.0.0.1:11434/api/generate",
    "http://127.0.0.1:11435/api/generate",
    "http://127.0.0.1:11436/api/generate",
    "http://127.0.0.1:11437/api/generate",
]
DEFAULT_INPUT = "data/raw/desensitized_texts_only.csv"
DEFAULT_OUTPUT_DIR = "data/processed/annotation"
MAX_WORKERS = 16
BATCH_SIZE = 20
MAX_RETRIES = 2
SAMPLE_SIZE = None  # 测试填 100；全量置 None


# ===================== 一级标签分类提示词 =====================
CLASSIFY_PROMPT = """你是医院客服文本标注专家。请对单条患者咨询文本选择唯一最匹配一级标签。

【20个互斥一级标签】
门诊服务, 住院服务, 急诊服务, 医保服务, 药品服务, 检验检查服务, 手术麻醉服务, 康复护理服务, 健康管理服务, 远程医疗服务, 公共卫生服务, 医疗费用管理, 病案档案管理, 行政后勤管理, 人力资源管理, 信息化管理, 质量安全管理, 便民服务, 投诉建议管理, 其他

【标签释义】
- 门诊服务：挂号、预约、就诊流程、科室选择、门诊缴费、改期取消、门诊药房
- 住院服务：入院手续、床位安排、住院缴费、出院流程、转院手续
- 急诊服务：急诊流程、急诊挂号、急诊等候
- 医保服务：医保政策、报销比例、异地医保、医保卡问题、慢病报销
- 药品服务：药品咨询、药品价格、药品购买、处方药、药品配送
- 检验检查服务：CT、B超、核磁共振、血常规、病理报告、检查预约
- 手术麻醉服务：手术预约、术前准备、麻醉咨询、术后恢复
- 康复护理服务：康复治疗、护理服务、康复训练
- 健康管理服务：体检、健康评估、预防保健、慢病管理
- 远程医疗服务：线上问诊、远程会诊、互联网医院
- 公共卫生服务：疫苗接种、传染病防控、妇幼保健、健康宣教
- 医疗费用管理：费用查询、欠费补缴、退款、费用明细
- 病案档案管理：病历复印、病案查询、病历修改
- 行政后勤管理：复印病历、盖章、停车、食堂、探视、陪护
- 人力资源管理：医生排班、专家出诊、人员信息
- 信息化管理：医院APP、微信公众号、挂号系统、报告查询系统
- 质量安全管理：医疗纠纷、患者安全、满意度调查、服务投诉
- 便民服务：楼层指引、电话查询、失物招领、便民措施
- 投诉建议管理：投诉、表扬、建议、意见反馈
- 其他：无法归类到以上任何标签的业务

【标注规则】
1. 每条文本仅输出1个标签，多选无效
2. 多业务混合取核心诉求对应的标签
3. 无法明确归类统一选“其他”
4. 禁止解释、禁止多余文字，只输出JSON

输出固定格式：
{"label": "选中的标签名称"}
"""


def build_single_text_prompt(text: str) -> str:
    return f"待标注患者咨询原文：{text}\n严格按照规则输出JSON标签结果。"


# ===================== 接口调用（多端口轮询） =====================
lock = Lock()
stats = {
    "total_requests": 0,
    "success_requests": 0,
    "failed_requests": 0,
    "empty_responses": 0,
    "parse_errors": 0,
}


def call_ollama_annotate(text: str, endpoint_cycle, model_name: str, retry_count: int = 0) -> dict:
    url = next(endpoint_cycle)
    full_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{CLASSIFY_PROMPT}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{build_single_text_prompt(text)}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>"
    )
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 128,
            "top_p": 0.9,
            "stop": ["<|eot_id|>"],
        },
    }
    with lock:
        stats["total_requests"] += 1
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        content = response.json().get("response", "").strip()
        if not content:
            with lock:
                stats["empty_responses"] += 1
            if retry_count < MAX_RETRIES:
                time.sleep(2)
                return call_ollama_annotate(text, endpoint_cycle, model_name, retry_count + 1)
            return {"label": "", "reason": "模型返回为空"}
        # 去除 markdown 代码块标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        try:
            parsed = json.loads(content)
            with lock:
                stats["success_requests"] += 1
            if isinstance(parsed, dict) and "label" in parsed:
                return {"label": parsed.get("label", ""), "reason": ""}
            return {"label": "", "reason": "输出JSON缺少label字段"}
        except json.JSONDecodeError:
            with lock:
                stats["parse_errors"] += 1
            json_match = re.search(r"\{.*?\}", content)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, dict) and "label" in parsed:
                        return {"label": parsed.get("label", ""), "reason": ""}
                except Exception:
                    pass
            if retry_count < MAX_RETRIES:
                time.sleep(2)
                return call_ollama_annotate(text, endpoint_cycle, model_name, retry_count + 1)
            return {"label": "", "reason": "JSON解析失败"}
    except requests.exceptions.Timeout:
        if retry_count < MAX_RETRIES:
            time.sleep(3)
            return call_ollama_annotate(text, endpoint_cycle, model_name, retry_count + 1)
        with lock:
            stats["failed_requests"] += 1
        return {"label": "", "reason": "请求超时"}
    except Exception as e:  # noqa: BLE001
        with lock:
            stats["failed_requests"] += 1
        if retry_count < MAX_RETRIES:
            time.sleep(2)
            return call_ollama_annotate(text, endpoint_cycle, model_name, retry_count + 1)
        return {"label": "", "reason": f"异常: {str(e)[:30]}"}


# ===================== 批次 / 断点 / 保存 =====================
def process_batch(batch_texts, batch_indices, results_dict, existing_results, endpoint_cycle, model_name):
    for i, text in enumerate(batch_texts):
        idx = batch_indices[i]
        if text in existing_results:
            with lock:
                results_dict[idx] = existing_results[text]
        else:
            res = call_ollama_annotate(text, endpoint_cycle, model_name)
            with lock:
                results_dict[idx] = {
                    "original_text": text,
                    "label": res.get("label", ""),
                    "reason": res.get("reason", ""),
                }
    return len(batch_texts)


def save_checkpoint(processed_indices, checkpoint_file):
    checkpoint = {
        "processed_indices": list(processed_indices),
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def load_checkpoint(checkpoint_file):
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return set(json.load(f).get("processed_indices", []))
    except FileNotFoundError:
        return set()


def save_results(results_dict, output_path):
    if not results_dict:
        return
    rows = [results_dict[i] for i in sorted(results_dict.keys())]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


# ===================== 主流程 =====================
def main():
    ap = argparse.ArgumentParser(description="单模型一级标签标注")
    ap.add_argument("--model", default="qwen3:latest", help="Ollama 模型名")
    ap.add_argument("--endpoints", nargs="*", default=DEFAULT_ENDPOINTS, help="Ollama /api/generate 端点列表")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="脱敏文本 CSV（含文本列）")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    ap.add_argument("--sample", type=int, default=SAMPLE_SIZE, help="仅处理前 N 条（测试用）")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = args.model.replace(":", "_").replace("/", "_")
    output_path = os.path.join(args.output_dir, f"annotated_{suffix}.csv")
    checkpoint_file = os.path.join(args.output_dir, f"checkpoint_{suffix}.json")

    print(f"[INFO] 模型: {args.model} | 输出: {output_path}")
    print(f"[INFO] 端点: {args.endpoints}")

    if not os.path.exists(args.input):
        print(f"[ERR] 输入文件不存在: {args.input}")
        return

    df_raw = pd.read_csv(args.input, encoding="utf-8-sig")
    print(f"[INFO] 原始数据 {len(df_raw)} 条")
    if args.sample and len(df_raw) > args.sample:
        df_raw = df_raw.head(args.sample)
        print(f"[INFO] 测试模式：仅前 {args.sample} 条")

    text_col = next((c for c in ["desensitized_text", "text", "query"] if c in df_raw.columns), None)
    if text_col is None:
        print(f"[ERR] 未找到文本列，现有列：{df_raw.columns.tolist()}")
        return
    texts = df_raw[text_col].astype(str).tolist()
    total = len(texts)

    processed = load_checkpoint(checkpoint_file)
    print(f"[INFO] 已完成 {len(processed)} 条，剩余 {total - len(processed)} 条")

    history = {}
    if os.path.exists(output_path):
        for _, row in pd.read_csv(output_path, encoding="utf-8-sig").iterrows():
            history[row["original_text"]] = {
                "original_text": row["original_text"],
                "label": row.get("label", ""),
                "reason": row.get("reason", ""),
            }
        print(f"[INFO] 加载历史结果 {len(history)} 条")

    endpoint_cycle = itertools.cycle(args.endpoints)
    pending = [(i, texts[i]) for i in range(total) if i not in processed]
    if not pending:
        print("[OK] 全部标注完成！")
        return

    batches = [
        ([x[0] for x in chunk], [x[1] for x in chunk])
        for chunk in (pending[s : s + args.batch] for s in range(0, len(pending), args.batch))
    ]
    print(f"[INFO] 共 {len(batches)} 批次，单批 {args.batch} 条")

    store = {}
    finished = set(processed)
    start_ts = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for b_idx, (idxs, txts) in enumerate(batches):
            fut = executor.submit(process_batch, txts, idxs, store, history, endpoint_cycle, args.model)
            fut.result()
            finished.update(idxs)
            save_checkpoint(finished, checkpoint_file)
            save_results(store, output_path)
            cost = time.time() - start_ts
            speed = len(finished) / cost * 60 if cost > 0 else 0
            print(f"[INFO] 进度 {len(finished)}/{total} ({len(finished)/total*100:.1f}%) | {speed:.1f}条/分钟")

    save_results(store, output_path)
    final = pd.read_csv(output_path, encoding="utf-8-sig")
    from collections import Counter
    cnt = Counter(final["label"].tolist())
    valid = len(final[final["label"] != ""])
    print(f"[OK] 完成：成功 {valid}/{len(final)} ({valid/len(final)*100:.1f}%)")
    for lab, c in cnt.most_common():
        if lab.strip():
            print(f"    {lab}: {c}")


if __name__ == "__main__":
    main()
