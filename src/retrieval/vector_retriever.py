from typing import Any, Dict, List, Union

from langchain_core.documents import Document

from src.retrieval.base_retriever import BaseRetriever
from src.rag import build_or_load_faiss_index


class VectorRetriever(BaseRetriever):
    """
    FAISS + OpenAI 임베딩 기반 벡터 검색기.
    - 인덱스는 build_or_load_faiss_index로 관리 (존재 시 로드, 실패 시 재빌드, rebuild 플래그 지원)
    - 기본 k=5
    - 단일 쿼리 문자열을 입력받아 상위 문서 리스트를 반환
    """

    def __init__(
        self,
        corpus_path: str = "src/corpus/corpus.json",
        index_dir: str = "faiss_index",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
        rebuild: bool = False,
        **kwargs: Any,
    ):
        super().__init__(data_path=corpus_path, **kwargs)
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.rebuild = rebuild

        self.vector_store = None

    def build_index(self):
        """FAISS 인덱스를 로드하거나 빌드합니다."""
        self.vector_store = build_or_load_faiss_index(
            corpus_path=self.data_path,
            index_dir=self.index_dir,
            rebuild=self.rebuild,
            embedding_model=self.embedding_model,
        )

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
            raise ValueError("VectorRetriever.retrieve currently supports a single query string.")

        if self.vector_store is None:
            self.build_index()

        k = top_k or self.top_k
        # similarity_search_with_score → [(Document, score)]
        results = self.vector_store.similarity_search_with_score(query, k=k)

        parsed = []
        for doc, score in results:
            meta = doc.metadata or {}
            parsed.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "title": meta.get("title"),
                    "score": score,
                    "metadata": meta,
                    "content": doc.page_content,
                }
            )
        return parsed


__all__ = ["VectorRetriever"]
