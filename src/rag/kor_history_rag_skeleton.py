"""
Skeleton architecture for a Korean history CSAT RAG system.

Constraints:
- Retrieval is only used to surface contextual hints for source-analysis questions.
- Retrieval must NOT be used for ordering/chronology questions.
- Uses FAISS IndexFlatIP with a fixed top-k=1 for the initial design.
- No real corpus or historical content is included; hook points are marked with TODOs.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "FAISS is required for this skeleton. Install faiss-cpu or faiss-gpu before running."
    ) from exc


# ----------------------------- Query classifier ----------------------------- #
class QueryClassifier:
    """
    Rule-based classifier that toggles RAG only for source-analysis questions.

    The classifier intentionally avoids any ML dependency. Keywords are minimal
    and can be extended as the team refines the heuristics.
    """

    SOURCE_KEYWORDS = ("사료", "자료", "발췌", "내용", "이 글", "다음 글")
    ORDERING_KEYWORDS = ("순서", "시기", "연대", "배열", "나열", "정렬", "전후", "차례", "먼저")

    def classify(self, query: str) -> str:
        """
        Classify a query into "source" (RAG ON) or "ordering" (RAG OFF).

        Args:
            query: Raw exam question text.

        Returns:
            Literal string "source" or "ordering".
        """
        normalized = query.replace(" ", "")
        if self._is_ordering_question(normalized):
            return "ordering"
        if self._is_source_question(normalized):
            return "source"
        # Default to conservative RAG OFF to avoid accidental use on ordering tasks.
        return "ordering"

    def _is_ordering_question(self, query: str) -> bool:
        return any(keyword in query for keyword in self.ORDERING_KEYWORDS)

    def _is_source_question(self, query: str) -> bool:
        return any(keyword in query for keyword in self.SOURCE_KEYWORDS)


# ----------------------------- Embedding layer ------------------------------ #
class DeterministicEmbedder:
    """
    Placeholder embedder with a stable, deterministic output.

    Replace _encode_text with a real embedding model (e.g., locally hosted) when
    connecting to production. The same interface is used for documents and
    queries to guarantee alignment.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: Corpus or query texts.

        Returns:
            2D numpy array of shape (len(texts), dim).
        """
        vectors = np.vstack([self._encode_text(text) for text in texts])
        # Normalization keeps IndexFlatIP equivalent to cosine similarity.
        faiss.normalize_L2(vectors)
        return vectors.astype("float32")

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Deterministic hash-based vector for reproducible mocking.

        TODO: Swap this stub with a real embedding model; do not hardcode API keys.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dim, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm


# ------------------------------- FAISS index -------------------------------- #
class FaissIndex:
    """
    Thin FAISS wrapper dedicated to cosine similarity retrieval (IndexFlatIP).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []

    def build(self, doc_texts: Sequence[str], embedder: DeterministicEmbedder) -> None:
        """
        Build the index from document texts.

        Args:
            doc_texts: List of raw corpus documents.
            embedder: Embedder used for both documents and queries.
        """
        self.doc_texts = list(doc_texts)
        self.doc_ids = [f"doc-{i}" for i in range(len(doc_texts))]
        embeddings = embedder.embed(doc_texts)
        self.index.reset()
        self.index.add(embeddings)

    def search(
        self, query: str, embedder: DeterministicEmbedder, top_k: int = 1
    ) -> List[Tuple[str, str, float]]:
        """
        Search the index using a query string.

        Args:
            query: User/exam query text.
            embedder: Same embedder used for documents.
            top_k: Number of hits to return (default 1; fixed for initial design).

        Returns:
            List of tuples (doc_id, doc_text, score).
        """
        if top_k != 1:
            logging.debug("top_k=%s requested; initial design fixes k=1.", top_k)
        if not self.doc_ids:
            raise ValueError("Index is empty. Build the index before searching.")

        query_vector = embedder.embed([query])
        scores, indices = self.index.search(query_vector, top_k)
        hits: List[Tuple[str, str, float]] = []
        for rank, doc_idx in enumerate(indices[0]):
            if doc_idx == -1:
                continue
            doc_id = self.doc_ids[doc_idx]
            hits.append((doc_id, self.doc_texts[doc_idx], float(scores[0][rank])))
        return hits


# ------------------------------- RAG control -------------------------------- #
@dataclass
class RagResult:
    """Structured output for downstream logging and evaluation."""

    query_type: str
    rag_applied: bool
    retrieved_doc_id: Optional[str]
    retrieved_text: Optional[str]


class RagController:
    """
    Orchestrates classification and retrieval without generating answers.
    """

    def __init__(
        self,
        classifier: QueryClassifier,
        embedder: DeterministicEmbedder,
        index: FaissIndex,
        top_k: int = 1,
    ):
        self.classifier = classifier
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    def run(self, query: str) -> RagResult:
        """
        Execute the RAG flow for a single query.

        Returns:
            RagResult containing classification, RAG toggle, and retrieved context.
        """
        query_type = self.classifier.classify(query)
        rag_on = query_type == "source"
        logging.info("Problem type=%s | RAG applied=%s", query_type, rag_on)

        if not rag_on:
            return RagResult(
                query_type=query_type,
                rag_applied=False,
                retrieved_doc_id=None,
                retrieved_text=None,
            )

        hits = self.index.search(query, self.embedder, top_k=self.top_k)
        if not hits:
            logging.info("RAG ON but no documents returned.")
            return RagResult(
                query_type=query_type,
                rag_applied=True,
                retrieved_doc_id=None,
                retrieved_text=None,
            )

        top_doc_id, top_doc_text, score = hits[0]
        logging.info("Top-1 doc_id=%s | score=%.4f", top_doc_id, score)
        return RagResult(
            query_type=query_type,
            rag_applied=True,
            retrieved_doc_id=top_doc_id,
            retrieved_text=top_doc_text,
        )


# ----------------------------- Mock wiring/demo ----------------------------- #
def _build_mock_corpus() -> List[str]:
    """
    Construct a placeholder corpus for local wiring tests.

    TODO: Replace with real corpus ingestion pipeline when data becomes available.
    """
    return [
        "사료형 문제용 모의 문서 A - 실제 역사 정보로 교체 예정.",
        "사료형 문제용 모의 문서 B - 실제 역사 정보로 교체 예정.",
        "사료형 문제용 모의 문서 C - 실제 역사 정보로 교체 예정.",
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    classifier = QueryClassifier()
    embedder = DeterministicEmbedder(dim=128)
    index = FaissIndex(dim=embedder.dim)

    mock_docs = _build_mock_corpus()
    index.build(mock_docs, embedder)

    controller = RagController(classifier, embedder, index, top_k=1)

    sample_queries = [
        "다음 사료를 보고 설명한 시기의 사회상을 고르시오.",
        "다음 사건을 발생 순서대로 올바르게 배열하시오.",
        "자료를 읽고 해당 제도의 특징을 고르시오.",
    ]

    for query in sample_queries:
        result = controller.run(query)
        logging.info(
            "query='%s' | rag=%s | doc_id=%s",
            query,
            result.rag_applied,
            result.retrieved_doc_id,
        )
        if result.retrieved_text:
            print(f"[Retrieved Context] {result.retrieved_text}")
        else:
            print("[RAG OFF] No retrieval performed.")


if __name__ == "__main__":
    main()
