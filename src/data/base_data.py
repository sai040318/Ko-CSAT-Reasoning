from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datasets import Dataset, DatasetDict

class BaseDataset(ABC):
    """
    모든 데이터셋 클래스의 기본 추상 클래스.
    새로운 데이터셋을 만들 때 이 클래스를 상속받아 구현하세요.
    """

    def __init__(self, data_path: str, **kwargs):
        """
        Args:
            data_path (str): 데이터 파일 경로
            **kwargs: 추가 설정
        """
        self.data_path = data_path
        self.dataset = None

    @abstractmethod
    def load_data(self) -> DatasetDict:
        """
        raw 데이터를 로드하고 HuggingFace DatasetDict 형태로 반환해야 합니다.
        (예: csv 읽기, json 파싱 등)
        """
        pass

    @abstractmethod
    def preprocess(self, tokenizer: Any, **kwargs) -> DatasetDict:
        """
        모델 학습에 들어갈 형태로 데이터를 전처리(토큰화 등)합니다.
        
        Args:
            tokenizer: HuggingFace Tokenizer
            **kwargs: max_length 등의 추가 인자
        """
        pass
