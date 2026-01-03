from typing import Any, Dict, List, Optional, Union

from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever


def _normalize_scores(
    results: List[Dict[str, Any]],
    score_key: str,
    *,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """
    score_key로 점수를 추출해 0~1로 정규화한 딕셔너리(doc_id->score)를 반환.
    - higher_is_better=False인 경우(예: L2 distance) 1 - norm_val로 뒤집어 유사도처럼 사용.
    """
    scores = [r.get(score_key) for r in results if r.get(score_key) is not None]
    if not scores:
        return {}
    max_s = max(scores)
    min_s = min(scores)
    norm = {}
    for r in results:
        s = r.get(score_key)
        if s is None:
            continue
        if max_s == min_s:
            norm_val = 1.0
        else:
            norm_val = (s - min_s) / (max_s - min_s)
        if not higher_is_better:
            norm_val = 1.0 - norm_val
        norm[r["doc_id"]] = norm_val
    return norm


class EnsembleRetriever(BaseRetriever):
    """
    BM25 + Vector 가중합 앙상블 리트리버.
    - 각각 top_k_candidates를 가져온 뒤 점수를 0~1로 정규화
    - combined_score = w_bm25 * bm25_norm + w_vec * vec_norm (없는 점수는 0)
    - 최종 top_k 반환
    """

    def __init__(
        self,
        corpus_path: str = "src/corpus/corpus.json",
        bm25_k: int = 10,
        vec_k: int = 10,
        top_k: int = 5,
        weight_bm25: float = 0.5,
        weight_vec: float = 0.5,
        embedding_model: str = "text-embedding-3-small",
        rebuild_faiss: bool = False,
        **kwargs: Any,
    ):
        super().__init__(data_path=corpus_path, **kwargs)
        self.bm25 = BM25Retriever(corpus_path=corpus_path, top_k=bm25_k)
        self.vec = VectorRetriever(
            corpus_path=corpus_path,
            index_dir="faiss_index",
            embedding_model=embedding_model,
            top_k=vec_k,
            rebuild=rebuild_faiss,
        )
        self.top_k = top_k
        self.weight_bm25 = weight_bm25
        self.weight_vec = weight_vec

    def build_index(self):
        # 개별 리트리버가 내부에서 lazy build
        return

    def retrieve(self, query_or_dataset: Union[str, Any], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not isinstance(query_or_dataset, str):
            raise ValueError("EnsembleRetriever.retrieve supports a single query string.")

        k_final = top_k or self.top_k

        bm25_res = self.bm25.retrieve(query_or_dataset, top_k=self.bm25.top_k)
        vec_res = self.vec.retrieve(query_or_dataset, top_k=self.vec.top_k)

        bm25_norm = _normalize_scores(bm25_res, "score", higher_is_better=True)
        # FAISS similarity_search_with_score는 보통 거리(L2)가 낮을수록 좋음 → 뒤집어서 사용
        vec_norm = _normalize_scores(vec_res, "score", higher_is_better=False)

        combined: Dict[str, Dict[str, Any]] = {}

        def upsert(res_list: List[Dict[str, Any]], source: str):
            for r in res_list:
                doc_id = r["doc_id"]
                if doc_id not in combined:
                    combined[doc_id] = {
                        "doc_id": doc_id,
                        "title": r.get("title"),
                        "bm25_score": None,
                        "vec_score": None,
                        "metadata": r.get("metadata"),
                        "content": r.get("content"),
                    }
                if source == "bm25":
                    combined[doc_id]["bm25_score"] = r.get("score")
                else:
                    combined[doc_id]["vec_score"] = r.get("score")

        upsert(bm25_res, "bm25")
        upsert(vec_res, "vec")

        scored: List[Dict[str, Any]] = []
        for doc_id, item in combined.items():
            s_b = bm25_norm.get(doc_id, 0.0)
            s_v = vec_norm.get(doc_id, 0.0)
            combined_score = self.weight_bm25 * s_b + self.weight_vec * s_v
            item["combined_score"] = combined_score
            scored.append(item)

        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored[:k_final]


__all__ = ["EnsembleRetriever"]
