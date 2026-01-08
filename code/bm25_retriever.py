import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from rank_bm25 import BM25Okapi

from src.retrieval.base_retriever import BaseRetriever


def _build_augmented_text(item: Dict[str, Any]) -> str:
    """
    search_text에 title/aliases/rag_matching_keywords(quotes, related_terms)을 덧붙여
    검색용 텍스트를 만든다.
    """
    parts: List[str] = []
    search_text = item.get("search_text", "")
    if search_text:
        parts.append(str(search_text))

    title = item.get("title")
    if title:
        parts.append(str(title))

    aliases = item.get("aliases", [])
    if aliases:
        parts.extend([str(a) for a in aliases])

    keywords = item.get("rag_matching_keywords", {}) or {}
    quotes = keywords.get("quotes", [])
    related_terms = keywords.get("related_terms", [])
    if quotes:
        parts.extend([str(q) for q in quotes])
    if related_terms:
        parts.extend([str(t) for t in related_terms])

    return " ".join(parts).strip()


def _content_dict_to_markdown(content: Dict[str, Any]) -> str:
    """content 딕셔너리를 간단한 Markdown 문자열로 변환합니다."""
    lines: List[str] = []
    for key, value in content.items():
        header = f"## {key}".replace("_", " ")
        lines.append(header)
        if isinstance(value, list):
            lines.extend([f"- {item}" for item in value])
        else:
            lines.append(str(value))
        lines.append("")  # 섹션 구분용 빈 줄
    return "\n".join(lines).strip()


def _tokenize_ko(text: str) -> List[str]:
    """
    간단한 한국어/영문 토큰화.
    - 형태소 분석기 없이도 동작 (Hangul, 영문/숫자 토큰 단위)
    - 필요 시 더 정교한 토크나이저로 교체 가능
    """
    if not text:
        return []
    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", text)


class BM25Retriever(BaseRetriever):
    """
    BM25 기반 키워드 검색기.
    - corpus.json의 search_text를 사용하여 인덱스를 생성
    - 기본 k=5
    """

    def __init__(
        self,
        corpus_path: str = "src/corpus/corpus.json",
        top_k: int = 5,
        **kwargs: Any,
    ):
        super().__init__(data_path=corpus_path, **kwargs)
        self.top_k = top_k

        self.docs: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def _load_corpus(self):
        corpus_file = Path(self.data_path)
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        corpus = json.loads(corpus_file.read_text(encoding="utf-8"))
        self.docs = []
        self.tokenized_corpus = []

        for item in corpus:
            augmented_text = _build_augmented_text(item)
            tokens = _tokenize_ko(augmented_text)
            self.tokenized_corpus.append(tokens)

            metadata = {
                "doc_id": item["doc_id"],
                "title": item.get("title"),
                "category": item.get("category"),
                "aliases": item.get("aliases", []),
                "rag_matching_keywords": item.get("rag_matching_keywords", {}),
                "full_content": _content_dict_to_markdown(item.get("content", {})),
            }
            self.docs.append(
                {
                    "doc_id": item["doc_id"],
                    "title": item.get("title"),
                    "search_text": item.get("search_text", ""),
                    "augmented_text": augmented_text,
                    "metadata": metadata,
                }
            )

    def build_index(self):
        """BM25 인덱스를 생성합니다."""
        self._load_corpus()
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(
        self, query_or_dataset: Union[str, Any], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        단일 문자열 쿼리를 받아 상위 문서를 반환합니다.

        Returns:
            List[Dict]: [{doc_id, title, score, metadata, content}, ...]
        """
        if isinstance(query_or_dataset, str):
            query = query_or_dataset
        else:
            raise ValueError("BM25Retriever.retrieve currently supports a single query string.")

        if self.bm25 is None:
            self.build_index()

        k = top_k or self.top_k
        query_tokens = _tokenize_ko(query)
        scores = self.bm25.get_scores(query_tokens)

        # 상위 k 인덱스 추출
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[Dict[str, Any]] = []
        for idx in sorted_idx:
            doc = self.docs[idx]
            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "score": scores[idx],
                    "metadata": doc["metadata"],
                    "content": doc["augmented_text"],
                }
            )

        return results


__all__ = ["BM25Retriever"]
