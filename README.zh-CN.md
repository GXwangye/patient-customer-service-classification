# 多大型语言模型协同标注与 BERT 分类的医院患者客服提问分类

**一项开发与验证研究**｜方法代码与示例数据

[English documentation](README.md)

本仓库对应论文《多大型语言模型协同标注与 BERT 分类的医院患者客服提问分类：一项开发与验证研究》，收录方法本体代码与最小语料样本，覆盖「从大规模含噪患者客服文本到可训练、可审计业务标签」的完整链路。

每条脱敏文本由三个大语言模型（GLM、Qwen、LLaMA）**各自独立标注**，投票聚合后，未取得三模型一致者交由 DeepSeek 复核，再经人工核对形成确定标签。这些标签用于微调 `bert-base-chinese`，把标签知识固化为可批量推理的模型。

---

## 本仓库不含什么

| 项目 | 原因 |
|---|---|
| 全量语料 | 131,111 条唯一文本含患者健康信息，仅以脱敏抽样形式公开（下详） |
| 训练好的模型权重 | 约 409 MB，超出 GitHub 单文件上限，可向作者索取 |

---

## 处理链路

```
原始客服记录
  → preprocessing/clean.py        提取患者侧咨询
  → preprocessing/desensitize.py  替换直接身份信息
  → preprocessing/dedup.py        停词归一化、去重、全局编号 uid
  → annotation/annotate.py        三模型各自独立标注
  → annotation/distill.py         候选标签 → 20 个一级标签
  → classification/train.py       BERT 微调与评估
  → classification/predict.py     批量推理
```

各阶段语料量：

| 阶段 | 文本数 |
|---|---|
| 采集原始记录 | 152,369 |
| 可评估记录 | 151,969 |
| 清洗后有效咨询 | 145,128 |
| 剔除空值与纯寒暄后可分析 | 144,996 |
| 去重后唯一文本（分析宇宙） | **131,111** |

131,111 条唯一文本的标注结果：

| 票型 | 文本数 | 占比 |
|---|---|---|
| 三模型完全一致 | 74,606 | 56.9% |
| 2:1 多数 | 44,891 | 34.2% |
| 1:1:1 分歧 | 11,614 | 8.9% |

其中 96,104 条无需人工介入直接采纳，8,798 条经人工核对，合计 **104,902 条带标签文本**（80.0%）。余下 26,209 条（19.99%）仍属歧义，未进入训练。三模型一致性为 Fleiss κ = 0.6189。

---

## 目录结构

```
.
├── README.md                       英文说明
├── README.zh-CN.md                 本文件
├── LICENSE                         MIT
├── requirements.txt
├── preprocessing/
│   ├── clean.py                    咨询提取（本地 Ollama）
│   ├── desensitize.py              身份信息替换（本地 Ollama）
│   └── dedup.py                    停词归一化、去重、uid 编号
├── annotation/
│   ├── prompts.md                  三类标注提示词模板
│   ├── annotate.py                 单模型一级标签标注
│   └── distill.py                  标签蒸馏（候选 → 20 类）
├── classification/
│   ├── label_mapping.json          20 个一级标签 ↔ 编号
│   ├── train.py                    BERT 训练、内部测试与外部验证
│   └── predict.py                  批量或单条推理
├── stats/
│   └── aggregate_statistics.json   语料、标注、一致性与建模聚合统计
└── data/
    └── examples/
        └── example_texts.csv       250 条示例（id, label, label_id, text）
```

---

## 安装

```bash
pip install -r requirements.txt
```

预处理与标注步骤调用本地 **Ollama** 部署的模型，默认对 4 个端点（`11434`–`11437`）做轮询并发，可用 `--endpoints` 改为与自建环境一致。训练推荐 **GPU**，`bert-base-chinese` 首次运行时从 Hugging Face 自动下载。

---

## 快速开始

### 1. 提取患者咨询

```bash
python preprocessing/clean.py \
    --input  data/raw/raw_transcripts.csv \
    --output data/processed/cleaned.csv \
    --model  llama3.1:latest
```

输出列：`original_text, cleaned_text, is_valid, is_complete, reason`。

### 2. 替换身份信息

```bash
python preprocessing/desensitize.py \
    --input      data/processed/cleaned.csv \
    --output     data/processed/desensitized.csv \
    --texts-only data/processed/desensitized_texts_only.csv \
    --model      llama3.1:latest
```

提示词只替换四类明确指向个人身份的信息（姓名、电话号码、身份证号、病历号／住院号），并要求模型在不完全确定时保持原文不动。输出列：`original_text, desensitized_text, sensitive_types, reason`，其中 `sensitive_types` 为 JSON 数组，如 `["姓名"]` 或 `[]`。

### 3. 去重与全局编号

```bash
python preprocessing/dedup.py \
    --input  data/processed/desensitized_texts_only.csv \
    --output data/processed/dedup_clean_corpus.csv \
    --report data/processed/dedup_report.json
```

归一化按「短语优先、单字助词其次」的顺序剥离数据驱动的停词表（43 个短语 + 12 个单字），在归一化键上去重，并按首次出现顺序赋予 `uid`。剥离前后为空的文本一并剔除。**这一步正是 145,128 条有效咨询收敛为 131,111 条分析宇宙的环节。** 输出列：`uid, original_text, norm_key`。

### 4. 三模型独立标注

```bash
python annotation/annotate.py --model qwen3:latest \
       --input data/processed/desensitized_texts_only.csv \
       --output-dir data/processed/annotation
python annotation/annotate.py --model glm4:9b   --input ...
python annotation/annotate.py --model llama3.1  --input ...
```

三次运行互相独立，各自输出 `annotated_<模型>.csv`。所用提示词见 `annotation/prompts.md`。

### 5. 标签蒸馏

```bash
python annotation/distill.py --input data/processed/distill_input/pre_annotated_data.csv
```

将约 1,059 条候选业务描述蒸馏收敛为 `classification/label_mapping.json` 中的 20 个一级标签。

### 6. BERT 训练

```bash
python classification/train.py \
       --data     data/processed/voting_results.csv \
       --out      final_model \
       --external data/processed/external_validation.csv
```

输入需含 `original_text` 与 `final_label`。待定样本自动排除，空标签归入「其他」。二分类（门诊服务 vs 其他）使用同一脚本、改换标签列即可。类别权重默认关闭，因消融显示其未提升总体 F1。

### 7. 推理

```bash
python classification/predict.py --model-dir final_model \
       --input data/examples/example_texts.csv --out preds.csv

echo "星期六还能取药吗" | python classification/predict.py --model-dir final_model --text -
```

---

## 已报告性能

| 任务 | 数据划分 | 准确率 | F1 |
|---|---|---|---|
| 二分类（门诊服务 vs 其他） | 内部测试 | 0.9616 | 0.9486 |
| 二分类（门诊服务 vs 其他） | 外部验证 | 0.9547 | 0.9285 |
| 20 类 | 内部测试 | 0.9296 | 0.8707（宏） |
| 20 类 | 外部验证 | 0.9158 | 0.8367（宏） |

内部划分在 104,902 条带标签文本上按 8:2 分层（训练 83,921 / 测试 20,981）；外部验证集为后续时期的 2,828 条文本。更完整的计数见 `stats/aggregate_statistics.json`。

---

## 数据说明与隐私声明

- **示例数据。** `data/examples/example_texts.csv` 从脱敏语料中分层抽取 250 条，覆盖 20 个标签。每条在发布前重新检查残留地名、机构名、人名、日期与号码，并逐条人工通读。样本均为简短、非污名化的咨询片段，不含可直接识别个人的信息。
- **全量语料。** 不公开。131,111 条唯一文本含患者健康信息，受数据源机构隐私与数据安全政策约束。
- **使用许可。** 若需将本代码用于其他机构数据，请遵循所在机构的伦理与隐私规范，并重新执行脱敏与标注流程。

---

## 一级标签体系

语料标注为 20 个互斥的一级业务标签。下表中文名即 `classification/label_mapping.json` 中的键，英文名取自论文定义。

| 编号 | 标签（中文） | Label (en) |
|---|---|---|
| 0 | 人力资源管理 | Human Resources Management |
| 1 | 住院服务 | Inpatient Services |
| 2 | 便民服务 | Convenience Services |
| 3 | 信息化管理 | Information Systems Management |
| 4 | 健康管理服务 | Health Management Services |
| 5 | 公共卫生服务 | Public Health Services |
| 6 | 其他 | Other |
| 7 | 医保服务 | Medical Insurance Services |
| 8 | 医疗费用管理 | Medical Expense Management |
| 9 | 康复护理服务 | Rehabilitation and Nursing Services |
| 10 | 急诊服务 | Emergency Services |
| 11 | 手术麻醉服务 | Surgery and Anesthesia Services |
| 12 | 投诉建议管理 | Complaint and Suggestion Management |
| 13 | 检验检查服务 | Laboratory and Examination Services |
| 14 | 病案档案管理 | Medical Records and Archive Management |
| 15 | 药品服务 | Pharmacy Services |
| 16 | 行政后勤管理 | Administration and Logistics Management |
| 17 | 质量安全管理 | Quality and Safety Management |
| 18 | 远程医疗服务 | Telemedicine Services |
| 19 | 门诊服务 | Outpatient Services |

机器可读的权威映射以 `classification/label_mapping.json` 为准。

---

## 许可与引用

代码采用 **MIT License**。若本研究对您的工作有帮助，请引用对应论文。归档版本见 Zenodo：

- **概念 DOI（涵盖全部版本，恒指向最新版）：** [10.5281/zenodo.22143320](https://doi.org/10.5281/zenodo.22143320)

---

## 主要依赖

`pandas` · `numpy` · `requests` · `torch` · `transformers` · `scikit-learn` · `matplotlib` · `seaborn` · `tqdm`
