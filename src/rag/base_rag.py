from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from src.retrieval.base_retriever import BaseRetriever
from src.model.base_model import BaseModel

class BaseRAG(ABC):
    """
    RAG(Retrieval-Augmented Generation) 파이프라인의 기본 추상 클래스.
    검색기와 모델을 결합하여 답변을 생성하는 논리를 관리합니다.
    """

    def __init__(
        self, 
        retriever: Optional[BaseRetriever] = None, 
        model: Optional[BaseModel] = None, 
        **kwargs
    ):
        """
        Args:
            retriever (BaseRetriever): 문서 검색기 객체 (선택)
            model (BaseModel): 답변 생성 모델 객체
            **kwargs: 프롬프트 템플릿 등 추가 설정
        """
        self.retriever = retriever
        self.model = model

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """
        단일 질문에 대해 RAG 과정을 수행하고 최종 정답을 반환합니다.
        
        1. (필요 시) 검색 수행
        2. 프롬프트 생성 (Context + Query)
        3. 모델을 통한 정답 생성
        """
        pass

    @abstractmethod
    def run_batch(self, queries: List[str], **kwargs) -> List[str]:
        """여러 질문에 대해 일괄적으로 RAG 과정을 수행합니다."""
        pass
