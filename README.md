# 医院患者客服提问分类（多 LLM 协同标注 + BERT）

本仓库是论文 **《多大型语言模型协同标注与 BERT 分类的医院患者客服提问分类研究》** 的方法本体代码与最小示例数据，用于复现"从大规模含噪患者客服文本到可训练、可审计业务标签"的标注—训练—推理流水线。

核心思想：用多个相互独立的大模型（GLM / Qwen / LLaMA）作"共识 + 分歧"双信号源，经 DeepSeek 复核与人工审核形成确定标签，再以 `bert-base-chinese` 训练判别式分类器，将标签知识固化为可批量推理的模型。

> 本仓库**不含**原始数据集（约 14.5 万条，已发布至 Zenodo）与训练好的模型权重（约 409 MB，受 GitHub 单文件 100 MB 限制，另行托管）。示例数据仅 40 条（每类 2 条），见 `data/examples/`。

---

## 目录结构

```
patient-customer-service-classification/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── annotation/                 # 多 LLM 协同标注 + 标签蒸馏
│   ├── prompts.md              # 三类标注提示词模板（候选归纳/一级分类/低置信复核）
│   ├── annotate.py             # 单模型一级标签标注（轮询本地 Ollama 端点）
│   └── distill.py              # 标签蒸馏流水线（候选 → 一级/二级标签体系）
├── classification/             # BERT 训练与推理
│   ├── label_mapping.json      # 20 个一级标签 ↔ 编号
│   ├── train.py                # BERT 多分类训练（含内部测试 + 外部验证）
│   └── predict.py              # 推理脚本（单条 / 批量 CSV，输出 Top-K）
├── web/                        # 分类应用演示（离线模型封装为交互工作台）
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── data/
    └── examples/
        └── example_texts.csv   # 40 条示例（每类 2 条，已再脱敏）
```

---

## 安装

```bash
pip install -r requirements.txt
```

- 标注 / 蒸馏依赖本地 **Ollama** 部署的 GLM / Qwen / LLaMA / DeepSeek（4 卡并行时端点为 `11434–11437`）。
- 训练推荐 **GPU**；`bert-base-chinese` 默认从 Hugging Face 自动下载。

---

## 快速开始

### 1. 多 LLM 协同标注（每个模型各跑一次）

```bash
# 三个模型分别运行，仅改 --model
python annotation/annotate.py --model qwen3:latest \
       --input data/raw/desensitized_texts_only.csv \
       --output-dir data/processed/annotation
python annotation/annotate.py --model glm4:9b   --input ...
python annotation/annotate.py --model llama3.1  --input ...
```

输出 `data/processed/annotation/annotated_<model>.csv`（`original_text, label, reason`）。
三模型结果经投票聚合：完全一致直接采纳、2:1 多数暂采纳、1:1:1 进入 DeepSeek 复核与人工审核（复核脚本在此流水线中复用 `annotate.py` 的接口与 `prompts.md` 的 Prompt 3）。

### 2. 标签蒸馏（候选 → 规范标签体系）

```bash
python annotation/distill.py --input data/processed/distill_input/pre_annotated_data.csv
```

将 LLM 归纳的细粒度候选（约 1,059 个）经六节点蒸馏收敛为 20 个一级标签（`classification/label_mapping.json`）。

### 3. BERT 训练

```bash
python classification/train.py \
       --data data/processed/voting_results.csv \
       --out final_model \
       --external data/processed/external_validation.csv
```

- 输入需含 `original_text` 与 `final_label`；"待定"样本自动排除，空标签归入"其他"。
- 二分类（门诊服务 vs 其他）使用同一脚本、不同输入标签列即可。
- 类别权重默认关闭（论文消融显示其未提升总体 F1，仅改变精确率—召回权衡）。

### 4. 推理

```bash
python classification/predict.py --model-dir final_model \
       --input data/examples/example_texts.csv --out preds.csv
# 单条
echo "星期六还能取药吗" | python classification/predict.py --model-dir final_model --text -
```

### 5. Web 演示

`web/` 将离线模型封装为可交互分类工作台（输出 20 类一级标签、门诊服务二分类、Top-K 候选及置信度、建议流转部门）。本地起一个静态服务即可：

```bash
cd web && python -m http.server 8000   # 浏览器打开 http://127.0.0.1:8000
```

> 演示仅用于流程验证与内部评测；真实部署效果（工作量、响应时间、患者体验与误分流责任）尚未纳入前瞻性评估。

---

## 数据说明与隐私声明

- **示例数据** `data/examples/example_texts.csv`：从脱敏数据集（`desensitized_texts_only.csv`，依据 GB/T 39725-2020 在医院数据源头脱敏）中**真实抽取**每类 2 条，并在发布前**再次清洗**残留地名、院名、人名、日期与号码等准标识符，逐条人工复核。样本均为简短、非 stigmatizing 的咨询片段，不含可直接识别个人的信息。
- **原始数据集**（约 14.5 万条）体量较大且含患者健康信息，不随本仓库发布，已托管至 Zenodo（发布渠道同既有数据集）。
- **模型权重**（约 409 MB）不纳入本仓库，请通过 Zenodo / Hugging Face 链接获取（待补充）。
- 若需将本代码用于其他机构数据，请遵循所在机构的伦理与隐私规范，并重新执行脱敏与标注流程。

---

## 标签体系（20 个一级业务标签）

门诊服务、住院服务、急诊服务、医保服务、药品服务、检验检查服务、手术麻醉服务、康复护理服务、健康管理服务、远程医疗服务、公共卫生服务、医疗费用管理、病案档案管理、行政后勤管理、人力资源管理、信息化管理、质量安全管理、便民服务、投诉建议管理、其他。

---

## 许可与引用

- 代码采用 **MIT License**。
- 若本研究对您的工作有帮助，请引用对应论文（录用后补充 DOI / 期刊信息）。

---

## 主要依赖

`pandas` · `numpy` · `requests` · `torch` · `transformers` · `scikit-learn` · `matplotlib` · `seaborn` · `tqdm`
