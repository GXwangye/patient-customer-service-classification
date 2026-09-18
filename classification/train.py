"""
train.py — BERT 多分类训练（20 个一级业务标签）

使用 bert-base-chinese 作为判别式分类器，监督标签来自多 LLM 投票 +
DeepSeek 复核 + 人工审核后的确定标签（final_label）。脚本同时完成内部测试
评估与外部验证（若存在），并保存模型、标签映射与指标报告。

依赖：pip install -r requirements.txt  +  GPU 推荐
运行：python train.py --data data/processed/voting_results.csv --out final_model
说明：
  - 输入 CSV 需含 original_text 与 final_label 两列；"待定"样本被排除，空标签归入"其他"。
  - 二分类（门诊服务 vs 其他）使用同一脚本、不同输入标签列即可，详见 README。
"""

import os
import json
import argparse
import pickle
import warnings

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="评估"):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            p = torch.softmax(out.logits, dim=1)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels.extend(batch["labels"].numpy())
            probs.extend(p.cpu().numpy())
    return preds, labels, np.array(probs)


def main():
    ap = argparse.ArgumentParser(description="BERT 多分类训练")
    ap.add_argument("--data", required=True, help="含 original_text, final_label 的 CSV")
    ap.add_argument("--out", default="final_model", help="模型输出目录")
    ap.add_argument("--external", default=None, help="外部验证 CSV（含 original_text, final_label）")
    ap.add_argument("--model-name", default="bert-base-chinese")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--class-weights", action="store_true", help="启用类别权重（论文中未提升总体F1）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[INFO] 设备: {device}")

    df = pd.read_csv(args.data, encoding="utf-8-sig")
    df = df[df["original_text"].notna()]
    df["original_text"] = df["original_text"].astype(str)
    df = df[df["original_text"].str.strip() != ""]
    df["final_label"] = df["final_label"].fillna("其他").replace("", "其他")
    df = df[df["final_label"] != "待定"]
    print(f"[INFO] 有效样本: {len(df)} 条")

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["final_label"])
    num_labels = len(le.classes_)
    label_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    write_json(os.path.join(args.out, "label_mapping.json"), label_mapping)

    X = df["original_text"].tolist()
    y = df["label_encoded"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=RANDOM_SEED, stratify=y)
    print(f"[INFO] 训练 {len(X_tr)} / 测试 {len(X_te)} / 类别 {num_labels}")

    tok = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels).to(device)

    if args.class_weights:
        cw = torch.tensor(
            compute_class_weight("balanced", classes=np.arange(num_labels), y=y_tr),
            dtype=torch.float,
        ).to(device)
        print("[INFO] 已启用类别权重")
    else:
        cw = None

    tr_loader = DataLoader(MedicalDataset(X_tr, y_tr, tok, args.max_length), batch_size=args.batch_size, shuffle=True)
    te_loader = DataLoader(MedicalDataset(X_te, y_te, tok, args.max_length), batch_size=args.batch_size, shuffle=False)

    opt = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(args.epochs):
        model.train()
        tot_loss, correct, total = 0, 0, 0
        bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in bar:
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = out.loss
            tot_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            preds = torch.argmax(out.logits, dim=1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += batch["labels"].size(0)
            bar.set_postfix(loss=loss.item(), acc=correct / total)
        print(f"[INFO] Epoch {epoch+1}: loss={tot_loss/len(tr_loader):.4f} acc={correct/total:.4f}")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "label_encoder.pkl"), "wb") as handle:
        pickle.dump(le, handle)
    print(f"[OK] 模型已保存: {args.out}")

    # 内部测试
    pr, la, _ = evaluate(model, te_loader)
    int_metrics = {
        "accuracy": accuracy_score(la, pr),
        "f1_macro": f1_score(la, pr, average="macro"),
        "f1_weighted": f1_score(la, pr, average="weighted"),
        "precision_macro": precision_score(la, pr, average="macro"),
        "recall_macro": recall_score(la, pr, average="macro"),
        "test_size": len(la),
    }
    print(f"[内部测试] 准确率 {int_metrics['accuracy']:.4f} | 宏F1 {int_metrics['f1_macro']:.4f}")
    print(classification_report(la, pr, target_names=le.classes_, digits=4))
    write_json(os.path.join(args.out, "internal_metrics.json"), int_metrics)

    cm = confusion_matrix(la, pr)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Internal test confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_matrix.svg"), format="svg")
    plt.close()

    # 外部验证
    ext_metrics = None
    if args.external and os.path.exists(args.external):
        dfx = pd.read_csv(args.external, encoding="utf-8-sig")
        dfx["final_label"] = dfx["final_label"].fillna("其他").replace("", "其他")
        dfx = dfx[dfx["final_label"].isin(set(le.classes_))]
        dfx["e"] = le.transform(dfx["final_label"])
        ex_loader = DataLoader(
            MedicalDataset(dfx["original_text"].astype(str).tolist(), dfx["e"].tolist(), tok, args.max_length),
            batch_size=args.batch_size, shuffle=False,
        )
        prx, lax, _ = evaluate(model, ex_loader)
        ext_metrics = {
            "accuracy": accuracy_score(lax, prx),
            "f1_macro": f1_score(lax, prx, average="macro"),
            "f1_weighted": f1_score(lax, prx, average="weighted"),
            "test_size": len(lax),
        }
        print(f"[外部验证] 准确率 {ext_metrics['accuracy']:.4f} | 宏F1 {ext_metrics['f1_macro']:.4f}")
        write_json(os.path.join(args.out, "external_metrics.json"), ext_metrics)

    report = {
        "model_name": args.model_name,
        "num_labels": num_labels,
        "epochs": args.epochs,
        "internal": int_metrics,
        "external": ext_metrics,
        "label_mapping": label_mapping,
    }
    write_json(os.path.join(args.out, "experiment_report.json"), report)
    print("[OK] 训练完成，报告已保存。")


if __name__ == "__main__":
    main()
