import pandas as pd
from ast import literal_eval
from typing import Any, Dict, Optional
from datasets import Dataset, DatasetDict
from src.data.base_data import BaseDataset
from src.utils.registry import DATASET_REGISTRY

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
            full_inputs = []
            for p, q, q_plus, c in zip(examples['paragraph'], examples['question'], examples['question_plus'], examples['choices']):
                # 선택지 리스트를 문자열로 변환 (예: 1. ㄱ, 2. ㄴ ...)
                choices_str = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(c)])
                
                # 보기(question_plus) 처리
                q_plus_str = f"<보기>: {q_plus}" if q_plus and str(q_plus).strip() else ""
                
                # 최종 프롬프트 구성 (prompt_templates.py의 구조를 단순화하여 반영)
                text = f"지문:\n{p}\n\n질문: {q}\n{q_plus_str}\n\n선택지:\n{choices_str}\n\n정답:"
                full_inputs.append(text)

            model_inputs = tokenizer(
                full_inputs, 
                max_length=max_length, 
                truncation=True, 
                padding="max_length"
            )
            
            # 정답(레이블) 토큰화 (Generation 학습 시 필요)
            if 'answer' in examples and examples['answer'][0] is not None:
                labels = tokenizer(
                    [str(a) for d, a in zip(examples['id'], examples['answer'])],
                    max_length=8, # 정답은 보통 짧으므로
                    truncation=True,
                    padding="max_length"
                )
                model_inputs["labels"] = labels["input_ids"]
                
            return model_inputs

        # 데이터셋의 모든 샘플에 토큰화 적용
        processed_dataset = self.dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        return processed_dataset
