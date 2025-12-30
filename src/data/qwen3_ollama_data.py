"""
Qwen3 Ollama 모델 전용 데이터셋 클래스.

Ollama 기반 모델은 토크나이저가 필요 없으므로,
raw 데이터 형태 (paragraph, question, choices 등)를 그대로 반환합니다.
"""

import pandas as pd
from ast import literal_eval
from typing import Any, Optional
from datasets import Dataset, DatasetDict

from src.data.base_data import BaseDataset
from src.utils.registry import DATASET_REGISTRY


# TODO
"""
  문제 분석

  run.py가 cfg.model.max_seq_length 같은 HuggingFace 전용 설정을 직접 참조하고 있어서, Ollama 모델처럼 해당 설정이 없는 경우 오류 발생.

  해결 전략 3가지

  Option A: Config에 더미 값 추가 (run.py 수정 0)

  - qwen3_2507_thinking.yaml에 max_seq_length: 512 등 더미 값 추가
  - 단점: 불필요한 설정이 config에 들어감

  Option B: run.py에서 .get() 사용 (run.py 최소 수정)

  max_length=cfg.model.get("max_seq_length", 512),
  - 장점: 설정 없으면 기본값 사용, 기존 config 호환
  - 단점: run.py 수정 필요 (하지만 안전한 접근 방식이라 좋은 패턴)

  Option C: preprocess에 cfg 전체 전달 (인터페이스 변경)

  processed_dataset = dataset.preprocess(cfg=cfg)
  - 각 클래스가 필요한 것만 꺼내 씀
  - 단점: 기존 데이터셋 클래스 인터페이스 변경 필요

  ---
  추천: Option A (가장 깔끔)
"""


@DATASET_REGISTRY.register("qwen3-ollama")
class Qwen3OllamaDataset(BaseDataset):
    """
    Qwen3 Ollama 모델 전용 데이터셋.

    특징:
    - 토크나이저 없이 raw 데이터 반환
    - Ollama 모델의 predict 함수에서 직접 프롬프트 생성
    """

    def load_data(self) -> DatasetDict:
        """CSV 파일 로드 및 평탄화"""
        df = pd.read_csv(self.data_path)

        records = []
        for _, row in df.iterrows():
            problems = literal_eval(row["problems"])

            record = {
                "id": row["id"],
                "paragraph": row["paragraph"],
                "question": problems["question"],
                "choices": problems["choices"],
                "answer": problems.get("answer", None),
                "question_plus": row.get("question_plus", None),
            }

            # question_plus가 problems 안에 있는 경우도 처리
            if record["question_plus"] is None:
                record["question_plus"] = problems.get("question_plus", None)

            records.append(record)

        flattened_df = pd.DataFrame(records)
        dataset = Dataset.from_pandas(flattened_df)

        self.dataset = DatasetDict({"train": dataset})
        return self.dataset

    def preprocess(
        self,
        tokenizer: Any = None,  # Ollama는 토크나이저 불필요
        max_length: int = 512,
        template: str = "qwen3_2507_thinking",
        **kwargs,
    ) -> DatasetDict:
        """
        Ollama 모델용 전처리 (토큰화 없이 raw 데이터 반환)

        Args:
            tokenizer: 사용하지 않음 (None 허용)
            max_length: 사용하지 않음
            template: 프롬프트 템플릿 이름 (모델에서 사용)
            **kwargs: 추가 설정

        Returns:
            DatasetDict: raw 데이터셋 (paragraph, question, choices 등 포함)
        """
        if self.dataset is None:
            self.load_data()

        # Ollama 모델은 토큰화 없이 raw 데이터 그대로 사용
        # 템플릿 이름만 메타데이터로 저장
        processed_dataset = self.dataset

        # 템플릿 정보를 info에 저장 (선택적)
        if hasattr(processed_dataset, "info"):
            processed_dataset.info.description = f"Template: {template}"

        return processed_dataset


@DATASET_REGISTRY.register("qwen3-ollama-eval")
class Qwen3OllamaEvalDataset(Qwen3OllamaDataset):
    """
    Qwen3 Ollama 평가용 데이터셋.

    학습 데이터의 일부를 평가용으로 사용할 때 사용.
    answer 컬럼이 있는 데이터에 대해 예측 후 평가 가능.
    """

    def preprocess(
        self,
        tokenizer: Any = None,
        max_length: int = 512,
        template: str = "qwen3_2507_thinking",
        split_ratio: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> DatasetDict:
        """
        평가용 데이터셋 전처리 (train/test 분할 포함)

        Args:
            split_ratio: 테스트셋 비율 (기본값: 0.1)
            seed: 랜덤 시드

        Returns:
            DatasetDict: {"train": train_dataset, "test": test_dataset}
        """
        if self.dataset is None:
            self.load_data()

        # train/test 분할
        split_dataset = self.dataset["train"].train_test_split(
            test_size=split_ratio,
            seed=seed,
        )

        return DatasetDict(
            {
                "train": split_dataset["train"],
                "test": split_dataset["test"],
            }
        )
