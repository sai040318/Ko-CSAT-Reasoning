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

    # ==========================================================
    # 데이터 로드 (원본 그대로)
    # ==========================================================
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

    # ==========================================================
    # 전처리 (Gemma / Chat 모델 분기)
    # ==========================================================
    def preprocess(
        self,
        tokenizer: Any,
        max_length: int = 512,
        template: str = "base",
        add_generation_prompt: bool = False,
        filter_over_length: bool = False,
        truncation: bool = True,
        exclude_answer_from_prompt: bool = False,  # ← evaluate 모드용 추가
        **kwargs,
    ) -> DatasetDict:

        if self.dataset is None:
            self.load_data()

        is_gemma = "gemma" in (tokenizer.name_or_path or "").lower()

        def tokenize_fn(examples):
            # evaluate 모드에서는 answer를 프롬프트에서 제외
            examples_for_prompt = examples.copy() if exclude_answer_from_prompt else examples
            if exclude_answer_from_prompt:
                # answer를 None으로 설정하여 assistant 메시지가 추가되지 않도록
                examples_for_prompt["answer"] = [None] * len(examples["answer"])
            
            # 1. 공통: chat message 생성
            chat_messages = build_chat_messages(
                template_name=template,
                examples=examples_for_prompt,  # ← 수정된 examples 사용
            )

            # ------------------------------
            # Gemma 전용 경로
            # ------------------------------
            if is_gemma:
                prompts = [
                    build_gemma_prompt(
                        messages=msg,
                        add_generation_prompt=add_generation_prompt,
                    )
                    for msg in chat_messages
                ]

                return tokenizer(
                    prompts,
                    truncation=truncation,
                    max_length=max_length,
                    padding=False,
                )

            # ------------------------------
            # LLaMA / Qwen (기존 로직)
            # ------------------------------
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

        # 길이 필터링 (원본 그대로)
        if filter_over_length:
            original_len = len(processed_dataset["train"])
            processed_dataset = processed_dataset.filter(
                lambda x: len(x["input_ids"]) <= max_length
            )
            filtered_len = len(processed_dataset["train"])
            print(
                f"📊 데이터 필터링 완료: {original_len} -> {filtered_len} "
                f"(제외된 샘플 수: {original_len - filtered_len})"
            )

        return processed_dataset

# ==========================================================
# Gemma 전용 유틸 함수
# ==========================================================
def build_gemma_prompt(messages, add_generation_prompt: bool = False) -> str:
    """
    Gemma용 prompt 문자열 생성
    - system role 미지원 → user 메시지에 병합
    - chat_template 사용하지 않음
    """

    system_text = ""
    turns = []

    for m in messages:
        role = m["role"]
        content = m["content"].strip()

        if role == "system":
            system_text += content + "\n"

        elif role == "user":
            if system_text:
                content = system_text + content
                system_text = ""
            turns.append(
                f"<start_of_turn>user\n{content}<end_of_turn>\n"
            )

        elif role == "assistant":
            turns.append(
                f"<start_of_turn>model\n{content}<end_of_turn>\n"
            )

    if add_generation_prompt:
        turns.append("<start_of_turn>model\n")

    return "".join(turns)
