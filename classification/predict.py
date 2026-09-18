"""
predict.py — 使用训练好的 BERT 模型对客服文本做一级标签推理

依赖：pip install -r requirements.txt
运行：
  python predict.py --model-dir final_model --input data/examples/example_texts.csv --out preds.csv
  echo "星期六还能取药吗" | python predict.py --model-dir final_model --text -

输入：
  --input CSV 需含文本列（自动识别 text/original_text/query）；
  --text 直接传入单条文本（"-" 表示从 stdin 读取）。
输出：CSV 含 text, label_id, label, confidence, topk（Top-K 候选）。
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as handle:
        encoder = pickle.load(handle)
    return model, tokenizer, encoder


def predict_texts(model, tokenizer, encoder, texts, max_length=128, topk=3):
    encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        ).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    classes = encoder.classes_
    rows = []
    for i, p in enumerate(probs):
        order = np.argsort(p)[::-1]
        top = [(int(k), classes[k], float(p[k])) for k in order[:topk]]
        rows.append({
            "text": texts[i],
            "label_id": top[0][0],
            "label": top[0][1],
            "confidence": round(top[0][2], 4),
            "topk": "; ".join(f"{c}({prob:.3f})" for _, c, prob in top),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="BERT 一级标签推理")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", default=None, help="CSV 文件")
    ap.add_argument("--text", default=None, help="单条文本；'-' 读 stdin")
    ap.add_argument("--out", default="preds.csv")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    model, tokenizer, encoder = load_model(args.model_dir)

    if args.text is not None:
        texts = [sys.stdin.read().strip()] if args.text == "-" else [args.text]
    elif args.input:
        df = pd.read_csv(args.input, encoding="utf-8-sig")
        col = next((c for c in ["text", "original_text", "query"] if c in df.columns), None)
        if col is None:
            raise ValueError(f"未找到文本列，现有列：{df.columns.tolist()}")
        texts = df[col].astype(str).tolist()
    else:
        raise ValueError("请通过 --input 或 --text 提供输入")

    rows = predict_texts(model, tokenizer, encoder, texts, args.max_length, args.topk)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(out_df[["text", "label", "confidence", "topk"]].to_string(index=False))
    print(f"\n[OK] 预测结果已写入 {args.out}")


if __name__ == "__main__":
    main()
