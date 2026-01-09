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
    CSV 파일을 읽어 'problems' 컬럼을 파싱하고 평탄화합니다.
    """
    def load_data(self) -> DatasetDict:
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
                "question_plus": problems.get("question_plus", None),
            }
            records.append(record)

        flattened_df = pd.DataFrame(records)
        dataset = Dataset.from_pandas(flattened_df)

        self.dataset = DatasetDict({"train": dataset})
        return self.dataset

    def preprocess(
        self,
        tokenizer: Any,
        max_length: int = 512,
        template: str = "base",
        add_generation_prompt: bool = False,
        filter_over_length: bool = False,
        truncation: bool = True,
        exclude_answer_from_prompt: bool = False, 
        **kwargs,
    ) -> DatasetDict:

        if self.dataset is None:
            self.load_data()

        is_gemma = "gemma" in (tokenizer.name_or_path or "").lower()

        def tokenize_fn(examples):
            examples_for_prompt = examples.copy() if exclude_answer_from_prompt else examples
            if exclude_answer_from_prompt:
                examples_for_prompt["answer"] = [None] * len(examples["answer"])
            
            chat_messages = build_chat_messages(
                template_name=template,
                examples=examples_for_prompt, 
            )
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
                for msg in chat_messages
            ]

            return tokenizer(
                formatted_prompts,
                truncation=truncation,
                max_length=max_length,
                padding=False,
            )

        processed_dataset = self.dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=[
                col
                for col in self.dataset["train"].column_names
                if col not in ["id","answer"]
            ],
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )


        if filter_over_length:
            original_len = len(processed_dataset["train"])
            processed_dataset = processed_dataset.filter(
                lambda x: len(x["input_ids"]) <= max_length
            )
            filtered_len = len(processed_dataset["train"])

        return processed_dataset
