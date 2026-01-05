"""
FAISS 인덱스 빌드/로드 유틸리티.
- 기본 경로: ./faiss_index
- 정책: 존재하면 우선 로드, 실패 시 재빌드. --rebuild 플래그로 강제 재빌드 가능.
"""
import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def _build_augmented_text(item: Dict[str, Any]) -> str:
    """search_text에 title/aliases/rag_matching_keywords(quotes, related_terms)을 덧붙여 검색 텍스트 생성."""
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


def load_corpus_documents(corpus_path: str = "src/corpus/corpus.json") -> List[Document]:
    """corpus.json을 LangChain Document 리스트로 변환합니다."""
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    corpus = json.loads(corpus_file.read_text(encoding="utf-8"))
    documents: List[Document] = []

    for item in corpus:
        metadata = {
            "doc_id": item["doc_id"],
            "title": item.get("title"),
            "category": item.get("category"),
            "aliases": item.get("aliases", []),
            "rag_matching_keywords": item.get("rag_matching_keywords", {}),
            "full_content": _content_dict_to_markdown(item.get("content", {})),
        }
        augmented_text = _build_augmented_text(item)
        doc = Document(page_content=augmented_text, metadata=metadata)
        documents.append(doc)

    return documents


def build_or_load_faiss_index(
    corpus_path: str = "src/corpus/corpus.json",
    index_dir: str = "faiss_index",
    rebuild: bool = False,
    embedding_model: str = "text-embedding-3-small",
) -> FAISS:
    """
    FAISS 인덱스를 로드하거나 빌드합니다.
    - rebuild=True면 무조건 새로 빌드 후 저장.
    - rebuild=False면 index_dir이 있으면 로드, 실패 시 빌드 후 저장.
    """
    index_path = Path(index_dir)
    embeddings = OpenAIEmbeddings(model=embedding_model)

    if index_path.exists() and not rebuild:
        try:
            print(f"✅ Loading existing FAISS index from {index_path}")
            return FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:  # 로드 실패 시 재빌드
            print(f"⚠️ Failed to load existing FAISS index ({e}). Rebuilding...")

    print("🚧 Building FAISS index...")
    documents = load_corpus_documents(corpus_path)
    vector_store = FAISS.from_documents(documents, embeddings)

    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    print(f"💾 Saved FAISS index to {index_path}")
    return vector_store


__all__ = ["load_corpus_documents", "build_or_load_faiss_index"]
