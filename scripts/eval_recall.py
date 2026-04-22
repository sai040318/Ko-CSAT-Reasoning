"""
Recall@k evaluation script for Ko-CSAT-Reasoning.

Re-runs retrieval against the current corpus and computes Recall@1/3/5
using gold_keywords from the annotation sheet as ground truth.

Usage:
    python scripts/eval_recall.py
    python scripts/eval_recall.py --retriever ensemble
    python scripts/eval_recall.py --top_k 10 --output output/recall_results.csv
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def parse_gold_keywords(raw: str) -> Set[str]:
    """
    gold_keywords 파싱. 구분자로 ',' 또는 '/' 모두 허용.
    - "x" or "" → 빈 셋 (RAG 불가 또는 미라벨)
    - "doc_a,doc_b" or "doc_a / doc_b" → {"doc_a", "doc_b"}
    - "doc_a, x" → {"doc_a"}  (x는 제외)
    """
    raw = safe_text(raw)
    if not raw:
        return set()
    # '/' 와 ',' 둘 다 구분자로 처리
    normalized = raw.replace("/", ",")
    tokens = {t.strip() for t in normalized.split(",")}
    tokens.discard("x")
    return tokens


def is_rag_solvable(raw: str) -> bool:
    """gold_keywords에 유효한 doc_id가 하나라도 있으면 True."""
    return len(parse_gold_keywords(raw)) > 0


def hit_at_k(gold: Set[str], retrieved_ids: List[str], k: int) -> bool:
    return bool(gold & set(retrieved_ids[:k]))


def make_retriever(args: argparse.Namespace):
    if args.retriever == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever(corpus_path=args.corpus_path, top_k=args.top_k)

    if args.retriever == "vector":
        from src.retrieval.vector_retriever import VectorRetriever
        return VectorRetriever(
            corpus_path=args.corpus_path,
            index_dir=args.index_dir,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            rebuild=args.rebuild_faiss,
        )

    if args.retriever == "ensemble":
        from src.retrieval.ensemble_retriever import EnsembleRetriever
        return EnsembleRetriever(
            corpus_path=args.corpus_path,
            bm25_k=args.bm25_k,
            vec_k=args.vec_k,
            top_k=args.top_k,
            weight_bm25=args.weight_bm25,
            weight_vec=args.weight_vec,
            embedding_model=args.embedding_model,
            rebuild_faiss=args.rebuild_faiss,
        )

    raise ValueError(f"Unsupported retriever: {args.retriever}")


def main():
    parser = argparse.ArgumentParser(description="Compute Recall@k from annotation sheet.")
    parser.add_argument("--annotation", default="data/history_recall_annotation_sheet.csv")
    parser.add_argument("--corpus_path", default="src/corpus/corpus.json")
    parser.add_argument("--retriever", choices=["bm25", "vector", "ensemble"], default="bm25")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", default="output/recall_eval_results.csv")

    # vector / ensemble 전용
    parser.add_argument("--index_dir", default="faiss_index")
    parser.add_argument("--bm25_k", type=int, default=10)
    parser.add_argument("--vec_k", type=int, default=10)
    parser.add_argument("--weight_bm25", type=float, default=0.7)
    parser.add_argument("--weight_vec", type=float, default=0.3)
    parser.add_argument("--embedding_model", default="text-embedding-3-small")
    parser.add_argument("--rebuild_faiss", action="store_true")

    args = parser.parse_args()

    ann_df = pd.read_csv(args.annotation)
    retriever = make_retriever(args)

    ks = [k for k in [1, 3, 5] if k <= args.top_k]

    records = []
    hit_counts = {k: 0 for k in ks}
    total_rag = 0

    for _, row in ann_df.iterrows():
        raw_gold = safe_text(row.get("gold_keywords", ""))
        gold = parse_gold_keywords(raw_gold)
        solvable = is_rag_solvable(raw_gold)

        query = safe_text(row.get("query", ""))
        if not query:
            # query 컬럼이 없으면 paragraph + question으로 fallback
            query = "\n".join([
                safe_text(row.get("paragraph", "")),
                safe_text(row.get("question", "")),
            ]).strip()

        retrieved_ids: List[str] = []
        retrieved_titles: List[str] = []
        if solvable and query:
            results = retriever.retrieve(query, top_k=args.top_k)
            retrieved_ids = [safe_text(r.get("doc_id", "")) for r in results]
            retrieved_titles = [safe_text(r.get("title", "")) for r in results]

        record = {
            "id": safe_text(row.get("id", "")),
            "gold_keywords": raw_gold,
            "rag_solvable": solvable,
            "notes": safe_text(row.get("notes", "")),
        }

        if solvable:
            total_rag += 1
            for k in ks:
                hit = hit_at_k(gold, retrieved_ids, k)
                record[f"hit@{k}"] = hit
                if hit:
                    hit_counts[k] += 1
        else:
            for k in ks:
                record[f"hit@{k}"] = None  # 평가 대상 아님

        for i, (doc_id, title) in enumerate(zip(retrieved_ids, retrieved_titles), 1):
            record[f"retrieved_{i}_doc_id"] = doc_id
            record[f"retrieved_{i}_title"] = title

        records.append(record)

    out_df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    total = len(ann_df)
    skipped = total - total_rag

    print(f"\n{'='*45}")
    print(f"  Retriever : {args.retriever}")
    print(f"  Corpus    : {args.corpus_path}")
    print(f"  Total     : {total}  (RAG 평가 가능: {total_rag}, 제외: {skipped})")
    print(f"{'='*45}")
    for k in ks:
        recall = hit_counts[k] / total_rag if total_rag else 0.0
        print(f"  Recall@{k:<2} : {hit_counts[k]:3d} / {total_rag}  =  {recall:.1%}")
    print(f"{'='*45}")
    print(f"\n  상세 결과 저장: {args.output}\n")


if __name__ == "__main__":
    main()
