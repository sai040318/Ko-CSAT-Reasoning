import argparse
import math
import sys
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, List

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


def parse_problem(problem_value: Any) -> Dict[str, Any]:
    if isinstance(problem_value, dict):
        return problem_value
    if isinstance(problem_value, str):
        return literal_eval(problem_value)
    raise ValueError(f"Unsupported problems value: {type(problem_value)}")


def build_query(paragraph: str, question: str, choices: List[str], query_mode: str) -> str:
    paragraph = safe_text(paragraph)
    question = safe_text(question)
    choice_lines = [safe_text(choice) for choice in choices if safe_text(choice)]

    if query_mode == "question_only":
        return question
    if query_mode == "question_choices":
        return "\n".join([question, *choice_lines]).strip()
    if query_mode == "paragraph_question":
        return "\n".join([paragraph, question]).strip()
    if query_mode == "paragraph_question_choices":
        return "\n".join([paragraph, question, *choice_lines]).strip()

    raise ValueError(f"Unsupported query_mode: {query_mode}")


def make_retriever(args: argparse.Namespace):
    if args.retriever == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever

        return BM25Retriever(
            corpus_path=args.corpus_path,
            top_k=args.top_k,
        )
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
    if args.retriever == "wikipedia":
        from src.retrieval.wiki_retriever import WikipediaRetriever

        return WikipediaRetriever(
            lang=args.wiki_lang,
            top_k=args.top_k,
            timeout=args.wiki_timeout,
        )

    raise ValueError(f"Unsupported retriever: {args.retriever}")


def truncate_text(text: str, max_chars: int) -> str:
    text = safe_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def main():
    parser = argparse.ArgumentParser(
        description="Build a CSV annotation sheet for retrieval recall labeling."
    )
    parser.add_argument("--input", default="data/history_eval.csv")
    parser.add_argument("--output", default="output/history_recall_annotation_sheet.csv")
    parser.add_argument(
        "--retriever",
        choices=["bm25", "vector", "ensemble", "wikipedia"],
        default="ensemble",
    )
    parser.add_argument(
        "--query_mode",
        choices=[
            "question_only",
            "question_choices",
            "paragraph_question",
            "paragraph_question_choices",
        ],
        default="paragraph_question",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--id_contains", default="")

    parser.add_argument("--corpus_path", default="src/corpus/corpus.json")
    parser.add_argument("--index_dir", default="faiss_index")
    parser.add_argument("--embedding_model", default="text-embedding-3-small")
    parser.add_argument("--rebuild_faiss", action="store_true")

    parser.add_argument("--bm25_k", type=int, default=10)
    parser.add_argument("--vec_k", type=int, default=10)
    parser.add_argument("--weight_bm25", type=float, default=0.7)
    parser.add_argument("--weight_vec", type=float, default=0.3)

    parser.add_argument("--wiki_lang", default="ko")
    parser.add_argument("--wiki_timeout", type=int, default=10)

    parser.add_argument("--snippet_chars", type=int, default=240)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if args.id_contains:
        df = df[df["id"].astype(str).str.contains(args.id_contains, na=False)]
    if args.limit > 0:
        df = df.head(args.limit)

    retriever = make_retriever(args)
    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        problem = parse_problem(row["problems"])
        paragraph = safe_text(row.get("paragraph", ""))
        question = safe_text(problem.get("question", ""))
        choices = problem.get("choices", []) or []
        answer = safe_text(problem.get("answer", ""))

        query = build_query(
            paragraph=paragraph,
            question=question,
            choices=choices,
            query_mode=args.query_mode,
        )
        results = retriever.retrieve(query, top_k=args.top_k)

        record: Dict[str, Any] = {
            "id": safe_text(row.get("id", "")),
            "paragraph": paragraph,
            "question": question,
            "choices": " | ".join([safe_text(choice) for choice in choices]),
            "answer": answer,
            "query": query,
            "retriever": args.retriever,
            "query_mode": args.query_mode,
            "gold_titles": "",
            "gold_keywords": "",
            "notes": "",
        }

        for idx in range(args.top_k):
            key_idx = idx + 1
            if idx < len(results):
                result = results[idx]
                meta = result.get("metadata") or {}
                content = meta.get("full_content") or result.get("content", "")
                record[f"candidate_{key_idx}_doc_id"] = safe_text(result.get("doc_id", ""))
                record[f"candidate_{key_idx}_title"] = safe_text(result.get("title", ""))
                record[f"candidate_{key_idx}_score"] = result.get("score")
                record[f"candidate_{key_idx}_snippet"] = truncate_text(content, args.snippet_chars)
            else:
                record[f"candidate_{key_idx}_doc_id"] = ""
                record[f"candidate_{key_idx}_title"] = ""
                record[f"candidate_{key_idx}_score"] = ""
                record[f"candidate_{key_idx}_snippet"] = ""

        rows.append(record)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    print(f"Saved annotation sheet to: {output_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Retriever: {args.retriever}")
    print(f"Query mode: {args.query_mode}")


if __name__ == "__main__":
    main()
