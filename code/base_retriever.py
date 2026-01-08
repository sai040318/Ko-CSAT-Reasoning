from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import pandas as pd
from datasets import Dataset

class BaseRetriever(ABC):
    """
    모든 검색기(Retriever) 클래스의 기본 추상 클래스.
    TF-IDF, Dense(FAISS), Hybrid 검색기 등을 구현할 때 상속받습니다.
    """

    def __init__(self, data_path: str, **kwargs):
        """
        Args:
            data_path (str): 검색 대상 문서(예: wiki)가 포함된 데이터 경로
            **kwargs: 추가 설정
        """
        self.data_path = data_path

    @abstractmethod
    def build_index(self):
        """
        검색을 위한 인덱스를 생성하거나 로드합니다.
        (예: TF-IDF 행렬 계산, Vector DB 로드 등)
        """
        pass

    @abstractmethod
    def retrieve(self, query_or_dataset: Union[str, Dataset], top_k: int = 5) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        질문에 대해 관련성 높은 문서를 검색합니다.

        Args:
            query_or_dataset: 단일 질문(str) 또는 질문 데이터셋(Dataset)
            top_k: 반환할 상위 문서 개수

        Returns:
            - 단일 질문 시: List[Dict] (검색된 문서 리스트)
            - 데이터셋 시: pd.DataFrame (검색 결과가 추가된 데이터프레임)
        """
        pass
