from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datasets import Dataset


class BaseModel(ABC):
    """
    모든 모델 클래스의 기본 추상 클래스.
    팀원들이 새로운 모델(예: Gemma, Llama 등)을 실험할 때 이 클래스를 상속받습니다.
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Args:
            model_name_or_path (str): 사전학습된 모델 이름 또는 로컬 경로
            **kwargs: 추가 설정 (PEFT, LoRA 등)
        """
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def get_tokenizer(model_name_or_path: str, **kwargs):
        """
        모델에 맞는 토크나이저를 로드하고 반환합니다.
        필요한 경우 Chat Template 등의 설정을 이곳에서 수행합니다.
        """

        pass

    @abstractmethod
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, **kwargs):
        """
        모델 학습을 수행합니다.

        Args:
            train_dataset: 전처리된 학습 데이터셋
            eval_dataset: 전처리된 검증 데이터셋 (선택)
            **kwargs: 학습 하이퍼파라미터 (epochs, lr 등)
        """
        pass

    @abstractmethod
    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, float]:
        """
        모델 성능을 평가합니다.

        Args:
            dataset: 평가할 데이터셋
            **kwargs: 평가 설정

        Returns:
            Dict[str, float]: 평가 지표 (예: {"accuracy": 0.85, "f1": 0.80})
        """
        pass

    @abstractmethod
    def predict(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        주어진 데이터셋에 대해 추론을 수행합니다.

        Args:
            dataset: 전처리된 추론용 데이터셋
            **kwargs: 추론 설정 (max_new_tokens 등)

        Returns:
            Dict[str, Any]: {id: 예측값} 형태의 딕셔너리
        """
        pass

    @abstractmethod
    def save_model(self, save_path: str):
        """모델 및 토크나이저 저장"""
        pass

    @abstractmethod
    def load_model(self, load_path: str):
        """저장된 모델 및 토크나이저 로드"""
        pass
