"""
dedup.py — corpus de-duplication and global unique numbering (pipeline step 3).

This stage runs after cleaning and desensitization and before annotation, and it
depends on nothing downstream. Each de-identified text is normalised by removing
a data-driven stop-word list, and the normalised key is used both for
de-duplication and for the global identifier `uid` carried through the rest of
the pipeline.

Pipeline position
    raw transcripts -> clean.py -> desensitize.py -> dedup.py -> annotation/

Normalisation order matters. Phrases are stripped longest-first so that a short
stop word cannot consume part of a longer one, and single-character particles are
stripped only afterwards. Rows that become empty after stop-word removal carried
no analysable content (bare greetings or pure interjections) and are dropped.

Usage
    python preprocessing/dedup.py \
        --input  data/processed/desensitized_texts_only.csv \
        --output data/processed/dedup_clean_corpus.csv \
        --report data/processed/dedup_report.json

    # supply an alternative stop-word list
    python preprocessing/dedup.py --input ... --output ... --stopwords stopwords.txt

Output
    uid, original_text, norm_key — one row per unique text, uid starting at 1 and
    assigned in order of first appearance.

`norm_key` is the normalised key; join downstream annotation results back onto
uid by normalising their text column with the same function.
"""

import argparse
import csv
import json
import os
import re

csv.field_size_limit(10 ** 7)

STOP_PHRASES = [
    "医生您好", "您好医生", "大夫您好", "您好", "你好", "请问", "在吗", "在不在",
    "麻烦", "谢谢", "感谢", "打扰了", "我想咨询", "我想问", "我想了解", "我想", "我要",
    "想咨询", "想问", "想了解", "帮我", "帮忙", "咨询一下", "一下",
    "这个", "那个", "一个", "一些", "这种", "那种", "哪种",
    "怎么", "怎样", "怎么样", "如何", "什么", "哪个", "哪些", "哪里", "哪儿", "为什么", "为何", "多少",
]
STOP_PARTICLES = list("吗呢啊呀吧哦啦哟嘛的了么")

TEXT_COLUMNS = ["desensitized_text", "text", "cleaned_text", "query"]


def full_to_half(text):
    out = []
    for char in text:
        code = ord(char)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(char)
    return "".join(out)


def normalise(text, phrases, particles):
    """Stop-word-stripped key used for de-duplication and for uid back-joining."""
    if not isinstance(text, str):
        return ""
    text = full_to_half(text.lower())
    text = re.sub(r"\s+", "", text)
    for phrase in phrases:
        text = text.replace(phrase, "")
    for particle in particles:
        text = text.replace(particle, "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text)


def load_stopwords(path):
    """Read extra stop phrases from a file, one phrase per line."""
    with open(path, "r", encoding="utf-8") as handle:
        phrases = [line.strip() for line in handle if line.strip()]
    return phrases + STOP_PHRASES


def build_corpus(input_path, output_path, phrases, particles, text_column=None):
    phrases = sorted(phrases, key=len, reverse=True)
    uid_of_key, representative, order = {}, {}, []
    raw_total = empty_dropped = stopword_empty_dropped = 0

    with open(input_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        column = text_column or next(
            (c for c in TEXT_COLUMNS if c in (reader.fieldnames or [])), None
        )
        if column is None:
            raise SystemExit(f"no text column found; columns are {reader.fieldnames}")

        for row in reader:
            raw_total += 1
            text = (row.get(column) or "").strip()
            if not text:
                empty_dropped += 1
                continue
            key = normalise(text, phrases, particles)
            if not key:
                stopword_empty_dropped += 1
                continue
            if key not in uid_of_key:
                uid = len(order) + 1
                uid_of_key[key] = uid
                representative[uid] = text
                order.append((uid, text, key))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "original_text", "norm_key"])
        writer.writerows(order)

    report = {
        "input_rows": raw_total,
        "empty_dropped": empty_dropped,
        "stopword_empty_dropped": stopword_empty_dropped,
        "duplicate_dropped": raw_total - empty_dropped - stopword_empty_dropped - len(order),
        "unique_texts": len(order),
        "stop_phrases": len(phrases),
        "stop_particles": len(particles),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="De-duplicate the corpus and assign uids")
    parser.add_argument("--input", required=True, help="de-identified corpus CSV")
    parser.add_argument("--output", required=True, help="output CSV with uid, original_text, norm_key")
    parser.add_argument("--text-column", default=None, help="override automatic text-column detection")
    parser.add_argument("--stopwords", default=None, help="extra stop phrases, one per line")
    parser.add_argument("--report", default=None, help="optional JSON path for the removal report")
    args = parser.parse_args()

    phrases = load_stopwords(args.stopwords) if args.stopwords else list(STOP_PHRASES)
    report = build_corpus(args.input, args.output, phrases, STOP_PARTICLES, args.text_column)

    print(f"[INFO] input {report['input_rows']} rows")
    print(f"[INFO] empty dropped            : {report['empty_dropped']}")
    print(f"[INFO] empty after stop words   : {report['stopword_empty_dropped']}")
    print(f"[INFO] duplicates dropped       : {report['duplicate_dropped']}")
    print(f"[OK] unique texts (uid 1..{report['unique_texts']}) -> {args.output}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
