"""
Retrieval labeling helper.
- Query: question + choices + answer (oracle) to maximize recall.
- BM25: fast filter; if top1 is clearly higher than top2 (ratio), auto-assign.
- GPT-4o-mini: only for ambiguous cases; asks to pick the best doc_id(s) from top candidates.

Usage:
  python scripts/retrieval_labeling.py \
      --dataset src/data/history_dataset.csv \
      --corpus src/corpus/corpus.json \
      --output output/retrieval_labels.jsonl \
      --bm25_k 5 --vec_k 5 --ratio_thresh 1.5 --gpt_top 3
"""
import argparse
import json
import sys
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import BM25Retriever, VectorRetriever


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def get_answer_text(problem: Dict[str, Any]) -> str:
    """정답 인덱스(1~N)를 실제 선택지 텍스트로 치환"""
    choices = problem.get("choices", [])
    ans = problem.get("answer", "")
    ans_text = ""
    try:
        idx = int(ans) - 1
        if 0 <= idx < len(choices):
            ans_text = choices[idx]
    except Exception:
        pass
    return normalize_text(ans_text or ans)


def make_query(paragraph: str, problem: Dict[str, Any], answer_text: str) -> str:
    """
    Oracle query for labeling: paragraph + question + 정답 선택지 텍스트만 사용.
    (전체 선택지는 포함하지 않아 노이즈를 줄임)
    """
    q = problem.get("question", "")
    return normalize_text(f"{paragraph} {q} 정답: {answer_text}")



def load_dataset(path: Path) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        prob = literal_eval(r["problems"])
        rows.append(
            {
                "id": r["id"],
                "paragraph": normalize_text(r["paragraph"]),
                "problem": prob,
            }
        )
    return rows


def load_corpus(path: Path) -> Dict[str, Dict[str, Any]]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    by_id = {c["doc_id"]: c for c in corpus}
    return by_id


def ratio_confident(scores: List[float], ratio_thresh: float) -> bool:
    if len(scores) < 2:
        return True
    top1, top2 = scores[0], scores[1]
    if top2 <= 0:
        return top1 > 0
    return (top1 / top2) >= ratio_thresh


def call_gpt(client: OpenAI, query: str, candidates: List[Dict[str, Any]], max_select: int = 2) -> List[str]:
    """
    Ask GPT to choose up to max_select doc_ids from candidates.
    """
    cand_text = "\n".join(
        [
            f"- id: {c['doc_id']} | title: {c.get('title')} | snippet: {c.get('snippet','')}"
            for c in candidates
        ]
    )
    prompt = f"""
질문과 정답을 보고, 아래 후보 문서 중 근거가 될 가능성이 높은 문서 id를 최대 {max_select}개까지 골라 JSON 배열로만 답하세요.

질문+정답: {query}
후보:
{cand_text}

출력 예시: ["doc-001"] 또는 ["doc-001","doc-002"] 혹은 확신 없으면 [] 만.
"""
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )
    content = resp.output_text
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


def candidate_matches_answer(candidate: Dict[str, Any], answer_text: str) -> bool:
    """정답 텍스트가 후보 title/aliases/search_text/snippet 중 하나에 등장하면 True"""
    if not answer_text:
        return False
    answer_text = answer_text.lower()
    parts = []
    meta = candidate.get("metadata") or {}
    title = candidate.get("title") or meta.get("title") or ""
    parts.append(title)
    aliases = meta.get("aliases") or []
    parts.extend(aliases)
    parts.append(candidate.get("content", ""))
    parts.append(meta.get("full_content", ""))
    joined = " ".join([p for p in parts if p]).lower()
    return answer_text in joined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="src/data/history_dataset.csv")
    parser.add_argument("--corpus", default="src/corpus/corpus.json")
    parser.add_argument("--output", default="output/retrieval_labels.jsonl")
    parser.add_argument("--bm25_k", type=int, default=10)
    parser.add_argument("--vec_k", type=int, default=10)
    parser.add_argument("--ratio_thresh", type=float, default=1.5)
    parser.add_argument("--gpt_top", type=int, default=5, help="number of candidates to send to GPT when ambiguous")
    parser.add_argument("--max_select", type=int, default=2, help="max doc_ids GPT can return")
    parser.add_argument("--rebuild_faiss", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(dataset_path)
    corpus_by_id = load_corpus(corpus_path)

    bm25 = BM25Retriever(corpus_path=str(corpus_path), top_k=args.bm25_k)

    try:
        vec = VectorRetriever(
            corpus_path=str(corpus_path),
            index_dir="faiss_index",
            embedding_model="text-embedding-3-small",
            top_k=args.vec_k,
            rebuild=args.rebuild_faiss,
        )
    except Exception as e:
        print(f"⚠️ Vector retriever init failed, skipping dense search: {e}")
        vec = None

    try:
        client = OpenAI()
        client_responsive = True
    except Exception as e:
        print(f"⚠️ OpenAI client init failed, GPT fallback disabled: {e}")
        client_responsive = False

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            ans_text = get_answer_text(row["problem"])
            qtext = make_query(row["paragraph"], row["problem"], ans_text)

            bm25_results = bm25.retrieve(qtext, top_k=args.bm25_k)
            bm25_scores = [r["score"] for r in bm25_results]

            vec_results: List[Dict[str, Any]] = []
            if vec is not None:
                try:
                    vec_results = vec.retrieve(qtext, top_k=args.vec_k)
                except Exception as e:
                    print(f"⚠️ Dense retrieve failed for id={row['id']}: {e}")

            # 합치기: doc_id 기준으로 중복 제거 (BM25 우선, 이후 dense 점수 필드 추가)
            combined: Dict[str, Dict[str, Any]] = {}
            for r in bm25_results:
                combined[r["doc_id"]] = {
                    "doc_id": r["doc_id"],
                    "title": r.get("title"),
                    "bm25_score": r.get("score", 0.0),
                    "vec_score": None,
                    "snippet": r.get("content", ""),
                }
            for r in vec_results:
                if r["doc_id"] in combined:
                    combined[r["doc_id"]]["vec_score"] = r.get("score")
                else:
                    combined[r["doc_id"]] = {
                        "doc_id": r["doc_id"],
                        "title": r.get("title"),
                        "bm25_score": None,
                        "vec_score": r.get("score"),
                        "snippet": r.get("content", ""),
                    }

            # BM25 비율로 확신 판단
            confident = ratio_confident(bm25_scores, args.ratio_thresh)
            gold_ids: List[str] = []
            method = "auto_ratio"

            if confident and bm25_results:
                # 정답 텍스트가 후보에 포함되지 않으면 자동 확정 금지
                if ans_text and not candidate_matches_answer(bm25_results[0], ans_text):
                    confident = False
            if confident and bm25_results:
                gold_ids = [bm25_results[0]["doc_id"]]
            else:
                method = "gpt_fallback" if client_responsive else "uncertain"
                if client_responsive:
                    # 상위 gpt_top 후보만 전달 (BM25 우선 순위)
                    top_candidates = bm25_results[: args.gpt_top]
                    gold_ids = call_gpt(client, qtext, top_candidates, max_select=args.max_select)

            record = {
                "id": row["id"],
                "query": qtext,
                "gold_doc_ids": gold_ids,
                "method": method,
                "bm25_top": [
                    {
                        "doc_id": r["doc_id"],
                        "title": r.get("title"),
                        "score": float(r.get("score")) if r.get("score") is not None else None,
                    }
                    for r in bm25_results
                ],
                "vec_top": [
                    {
                        "doc_id": r["doc_id"],
                        "title": r.get("title"),
                        "score": float(r.get("score")) if r.get("score") is not None else None,
                    }
                    for r in vec_results
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved labels to {output_path}")


if __name__ == "__main__":
    main()
