# Multi-LLM Collaborative Annotation and BERT Classification for Hospital Patient Service Question Categorization

**a development and validation study** · method code and sample data

[中文说明 / Chinese documentation](README.zh-CN.md)

This repository accompanies the manuscript *Multi-LLM Collaborative Annotation and BERT Classification for Hospital Patient Service Question Categorization: a development and validation study*. It contains the method code and a minimal corpus sample, covering the path from noisy hospital patient-service transcripts to a trainable, auditable label system.

Each de-identified text is annotated independently by three language models (GLM, Qwen, LLaMA). The votes are aggregated, and the texts that do not reach three-model consensus are reviewed by DeepSeek and then verified by a human. The resulting labels train a `bert-base-chinese` classifier that turns the label system into a model which can be applied in batch.

---

## What this repository does not contain

| Item | Reason |
|---|---|
| The full corpus | 131,111 unique texts carrying patient health information; released only as the de-identified sample described below |
| Trained model weights | about 409 MB, above the GitHub per-file limit; available on request |

---

## Pipeline

```
raw transcripts
  → preprocessing/clean.py        extract the patient-side query
  → preprocessing/desensitize.py  replace directly identifying information
  → preprocessing/dedup.py        stop-word normalisation, de-duplication, uid
  → annotation/annotate.py        three models annotate each text independently
  → annotation/distill.py         candidate labels → 20 first-level labels
  → classification/train.py       BERT fine-tuning and evaluation
  → classification/predict.py     batch inference
```

Corpus counts at each stage:

| Stage | Texts |
|---|---|
| Raw records collected | 152,369 |
| Evaluable records | 151,969 |
| Valid queries after cleaning | 145,128 |
| Analysable after removing empty and greeting-only texts | 144,996 |
| Unique texts after de-duplication (the analysis corpus) | **131,111** |

Annotation outcome over the 131,111 unique texts:

| Vote pattern | Texts | Share |
|---|---|---|
| Unanimous (3 of 3) | 74,606 | 56.9% |
| Majority (2 of 1) | 44,891 | 34.2% |
| Split (1 each) | 11,614 | 8.9% |

Of these, 96,104 were accepted without human review and 8,798 were human-verified, giving **104,902 labelled texts** (80.0%). The remaining 26,209 (19.99%) stayed ambiguous and were not used for training. Agreement between the three models was Fleiss κ = 0.6189.

---

## Repository layout

```
.
├── README.md                       this file
├── README.zh-CN.md                 Chinese documentation
├── LICENSE                         MIT
├── requirements.txt
├── preprocessing/
│   ├── clean.py                    query extraction (local Ollama)
│   ├── desensitize.py              PII replacement (local Ollama)
│   └── dedup.py                    stop-word normalisation, de-duplication, uid
├── annotation/
│   ├── prompts.md                  the three annotation prompt templates
│   ├── annotate.py                 single-model first-level annotation
│   └── distill.py                  label distillation (candidates → 20 labels)
├── classification/
│   ├── label_mapping.json          the 20 first-level labels ↔ ids
│   ├── train.py                    BERT training, internal test, external validation
│   └── predict.py                  batch or single-text inference
├── stats/
│   └── aggregate_statistics.json   corpus, annotation, agreement and modelling counts
└── data/
    └── examples/
        └── example_texts.csv       250 sampled texts (id, label, label_id, text)
```

---

## Installation

```bash
pip install -r requirements.txt
```

The preprocessing and annotation steps call models served by a local **Ollama** instance. The default configuration polls four endpoints (`11434`–`11437`) concurrently; change `--endpoints` to match your deployment. Training expects a **GPU**; `bert-base-chinese` is downloaded from Hugging Face on first use.

---

## Quick start

### 1. Extract patient queries

```bash
python preprocessing/clean.py \
    --input  data/raw/raw_transcripts.csv \
    --output data/processed/cleaned.csv \
    --model  llama3.1:latest
```

Output columns: `original_text, cleaned_text, is_valid, is_complete, reason`.

### 2. Replace identifying information

```bash
python preprocessing/desensitize.py \
    --input      data/processed/cleaned.csv \
    --output     data/processed/desensitized.csv \
    --texts-only data/processed/desensitized_texts_only.csv \
    --model      llama3.1:latest
```

The prompt replaces only four categories of directly identifying information (personal name, phone number, national ID, medical-record number) and instructs the model to leave the text unchanged when it is not certain. Output columns: `original_text, desensitized_text, sensitive_types, reason`, where `sensitive_types` is a JSON array such as `["姓名"]` or `[]`.

### 3. De-duplicate and assign uids

```bash
python preprocessing/dedup.py \
    --input  data/processed/desensitized_texts_only.csv \
    --output data/processed/dedup_clean_corpus.csv \
    --report data/processed/dedup_report.json
```

Normalisation strips a data-driven stop-word list (43 phrases and 12 particles) longest-phrase-first, then de-duplicates on the normalised key and assigns `uid` in order of first appearance. Texts that are empty before or after stop-word removal are dropped. This step is what turns 145,128 valid queries into the 131,111-text analysis corpus. Output columns: `uid, original_text, norm_key`.

### 4. Annotate with three models

```bash
python annotation/annotate.py --model qwen3:latest \
       --input data/processed/desensitized_texts_only.csv \
       --output-dir data/processed/annotation
python annotation/annotate.py --model glm4:9b   --input ...
python annotation/annotate.py --model llama3.1  --input ...
```

Each run is independent and writes `annotated_<model>.csv`. The prompts used are documented in `annotation/prompts.md`.

### 5. Distil the label system

```bash
python annotation/distill.py --input data/processed/distill_input/pre_annotated_data.csv
```

Distils roughly 1,059 candidate descriptions into the 20 first-level labels listed in `classification/label_mapping.json`.

### 6. Train BERT

```bash
python classification/train.py \
       --data     data/processed/voting_results.csv \
       --out      final_model \
       --external data/processed/external_validation.csv
```

The input needs `original_text` and `final_label`; undetermined texts are dropped and empty labels fall back to "其他". The binary task (outpatient service vs other) uses the same script with a different label column. Class weighting is off by default, as it did not improve overall F1 in the ablation.

### 7. Inference

```bash
python classification/predict.py --model-dir final_model \
       --input data/examples/example_texts.csv --out preds.csv

echo "星期六还能取药吗" | python classification/predict.py --model-dir final_model --text -
```

---

## Reported performance

| Task | Split | Accuracy | F1 |
|---|---|---|---|
| Binary (outpatient vs other) | Internal test | 0.9616 | 0.9486 |
| Binary (outpatient vs other) | External validation | 0.9547 | 0.9285 |
| Twenty-class | Internal test | 0.9296 | 0.8707 (macro) |
| Twenty-class | External validation | 0.9158 | 0.8367 (macro) |

The internal split is 8:2 stratified (83,921 training / 20,981 test) over the 104,902 labelled texts. The external set is 2,828 texts from a later period. Further counts are in `stats/aggregate_statistics.json`.

---

## Data and privacy

- **Sample data.** `data/examples/example_texts.csv` holds 250 texts sampled from the de-identified corpus, stratified across the 20 labels. Before release each text was re-checked for residual place names, institution names, personal names, dates and numbers, and read through by hand. All samples are short, non-stigmatising service enquiries and contain no directly identifying information.
- **Full corpus.** Not released. The 131,111 unique texts carry patient health information and are covered by the data-source institution's privacy and security policy.
- **Licence of use.** Applying this code to another institution's data requires that institution's ethics and privacy clearance and a fresh run of the desensitisation and annotation steps.

---

## First-level label system

The corpus is annotated into 20 mutually exclusive first-level business labels. The Chinese names below are the keys used in `classification/label_mapping.json`; the English names are those defined in the manuscript.

| id | Label (zh) | Label (en) |
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

The authoritative machine-readable mapping is `classification/label_mapping.json`.

---

## Licence and citation

Code is released under the **MIT License**. Please cite the accompanying manuscript if this work is useful to you. The archived release is available through Zenodo:

- **Concept DOI (all versions, always resolves to the latest):** [10.5281/zenodo.22143320](https://doi.org/10.5281/zenodo.22143320)

---

## Dependencies

`pandas` · `numpy` · `requests` · `torch` · `transformers` · `scikit-learn` · `matplotlib` · `seaborn` · `tqdm`
