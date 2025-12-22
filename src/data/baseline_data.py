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
            problems = literal_eval(row['problems'])
            
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                'question_plus': problems.get('question_plus', None),
            }
            records.append(record)
        
        # 3. Pandas -> HuggingFace Dataset 변환
        flattened_df = pd.DataFrame(records)
        dataset = Dataset.from_pandas(flattened_df)
        
        # 4. DatasetDict 형태로 감싸서 반환 (나중에 train/val 분할을 위해)
        self.dataset = DatasetDict({"train": dataset})
        return self.dataset

    def preprocess(self, tokenizer: Any, max_length: int = 512, **kwargs) -> DatasetDict:
        """
        텍스트 데이터를 모델이 이해할 수 있는 토큰 ID로 변환합니다.
        (Generation Task를 위해 간단한 토큰화만 수행하거나, 
        SFT용 프롬프트를 구성하는 용도로 확장 가능합니다.)
        """
        if self.dataset is None:
            self.load_data()

        def tokenize_fn(examples):
            # 1. 데이터를 Chat Message 형식으로 변환
            chat_messages = build_chat_messages(
                #template_name=cfg.template,  # yaml / cli에서 지정
                template_name="base",  # yaml / cli에서 지정
                examples=examples,
            )

            # 2. Chat Template 적용 및 토큰화
            # apply_chat_template은 리스트의 리스트를 받으면 배치를 처리함
            formatted_prompts = [
                tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
                for msg in chat_messages
            ]
            
            model_inputs = tokenizer(
                formatted_prompts,
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )
            
            # SFTTrainer는 input_ids를 받아서 labels를 자동 생성(DataCollatorForCompletionOnlyLM 사용 시)
            # 여기서는 labels를 명시적으로 만들지 않고 input_ids만 반환해도 됨
            # (Trainer의 DataCollator가 처리)
                
            return model_inputs

        # 데이터셋의 모든 샘플에 토큰화 적용
        processed_dataset = self.dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        return processed_dataset
