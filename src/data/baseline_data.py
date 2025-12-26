import pandas as pd
from ast import literal_eval
from typing import Any, Dict, Optional
from datasets import Dataset, DatasetDict
from src.data.base_data import BaseDataset
from src.utils.registry import DATASET_REGISTRY
from prompt.prompt_templates import build_chat_messages


@DATASET_REGISTRY.register("baseline")
class BaselineDataset(BaseDataset):
    """
    대회에서 제공된 베이스라인 데이터 로더.
    CSV 파일을 읽어 'problems' 컬럼을 파싱하고 평탄화합니다.
    """

    def load_data(self) -> DatasetDict:
        """
        데이터를 로드하고 전처리 전의 DatasetDict를 반환합니다.
        """
        # 1. CSV 로드
        df = pd.read_csv(self.data_path)

        # 2. 'problems' 컬럼(문자열 형태의 JSON) 파싱 및 데이터 평탄화
        records = []
        for _, row in df.iterrows():
            # 문자열 형태의 딕셔너리를 실제 객체로 변환
            problems = literal_eval(row["problems"])

            record = {
                "id": row["id"],
                "paragraph": row["paragraph"],
                "question": problems["question"],
                "choices": problems["choices"],
                "answer": problems.get("answer", None),
                "question_plus": problems.get("question_plus", None),
            }
            records.append(record)

        # 3. Pandas -> HuggingFace Dataset 변환
        flattened_df = pd.DataFrame(records)
        dataset = Dataset.from_pandas(flattened_df)

        # 4. DatasetDict 형태로 감싸서 반환 (나중에 train/val 분할을 위해)
        self.dataset = DatasetDict({"train": dataset})
        return self.dataset

    def preprocess(
        self,
        tokenizer: Any,
        max_length: int = 512,
        template: str = "base",
        add_generation_prompt: bool = False,
        filter_over_length: bool = False,
        **kwargs,
    ) -> DatasetDict:
        """
        텍스트 데이터를 모델이 이해할 수 있는 토큰 ID로 변환합니다.

        Args:
            add_generation_prompt (bool): Inference 시 답변 생성을 위한 프롬프트 추가 여부
            filter_over_length (bool): max_length를 초과하는 데이터 필터링 여부
        """
        if self.dataset is None:
            self.load_data()

        def tokenize_fn(examples):
            # 1. 데이터를 Chat Message 형식으로 변환
            chat_messages = build_chat_messages(
                template_name=template,  # yaml / cli에서 지정
                examples=examples,
            )

            # 2. Chat Template 적용 및 토큰화
            # TODO: pd.iterrows() 가장 느린 이터레이션 df.apply 늼이 제일 나음
            # 병렬화 처리 필요, 이거 리스트 컴프리헨션
            # TODO: 리스트 컴프리헨션 이렇게 하면 엄청 느림, 오버헤드 꽤 큼
            formatted_prompts = [
                tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=add_generation_prompt)
                for msg in chat_messages
            ]

            model_inputs = tokenizer(
                formatted_prompts,
                truncation=False,  # Baseline처럼 truncation 없이 토크나이징 후 필터링
                padding=False,
            )

            return model_inputs

        # 데이터셋의 모든 샘플에 토큰화 적용
        processed_dataset = self.dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=[col for col in self.dataset["train"].column_names if col not in ["id"]],
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        # 3. 데이터 필터링 (설정에 따라 동작)
        if filter_over_length:
            original_len = len(processed_dataset["train"])
            processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
            filtered_len = len(processed_dataset["train"])
            print(
                f"📊 데이터 필터링 완료: {original_len} -> {filtered_len} (제외된 샘플 수: {original_len - filtered_len})"
            )

        return processed_dataset
