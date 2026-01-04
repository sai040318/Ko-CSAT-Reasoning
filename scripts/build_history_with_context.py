"""
history_dataset.csv를 gold_doc_ids 기반 CONTEXT를 붙여 확장하는 스크립트.
- 입력: history_dataset.csv, retrieval_labels.jsonl, corpus.json
- 출력: history_with_context.csv (기존 컬럼 동일: id, paragraph, problems, question_plus)
- paragraph 뒤에 [CONTEXT] 섹션으로 gold_doc_ids에 해당하는 문서 내용을 붙임.

사용 예:
python scripts/build_history_with_context.py \
  --dataset src/data/history_dataset.csv \
  --labels output/retrieval_labels.jsonl \
  --corpus src/corpus/corpus.json \
  --output output/history_with_context.csv
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def content_dict_to_markdown(content: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key, value in content.items():
        header = f"## {key}".replace("_", " ")
        lines.append(header)
        if isinstance(value, list):
            lines.extend([f"- {item}" for item in value])
        else:
            lines.append(str(value))
        lines.append("")  # 섹션 구분
    return "\n".join(lines).strip()


def load_corpus(path: Path) -> Dict[str, Dict[str, Any]]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    by_id = {}
    for item in corpus:
        md = content_dict_to_markdown(item.get("content", {}))
        by_id[item["doc_id"]] = {
            "title": item.get("title", ""),
            "content_md": md,
        }
    return by_id


def load_labels(path: Path) -> Dict[str, List[str]]:
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        mapping[rec["id"]] = rec.get("gold_doc_ids") or []
    return mapping


def build_context(doc_ids: List[str], corpus: Dict[str, Dict[str, Any]]) -> str:
    parts: List[str] = []
    for did in doc_ids:
        doc = corpus.get(did)
        if not doc:
            continue
        parts.append(f"[{did}] {doc['title']}\n{doc['content_md']}")
    return "\n\n".join(parts).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="src/data/history_dataset.csv")
    parser.add_argument("--labels", default="output/retrieval_labels.jsonl")
    parser.add_argument("--corpus", default="src/corpus/corpus.json")
    parser.add_argument("--output", default="src/data/history_with_context.csv")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    labels_path = Path(args.labels)
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(corpus_path)
    labels = load_labels(labels_path)

    df = pd.read_csv(dataset_path)
    new_rows = []
    for _, row in df.iterrows():
        rid = row["id"]
        gold_ids = labels.get(rid, [])
        context = build_context(gold_ids, corpus)
        paragraph = row["paragraph"]
        if context:
            paragraph = f"{paragraph}\n\n[CONTEXT]\n{context}"
        new_rows.append(
            {
                "id": rid,
                "paragraph": paragraph,
                "problems": row["problems"],
                "question_plus": row.get("question_plus", ""),
            }
        )

    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
