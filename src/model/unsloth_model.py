import torch
import numpy as np
from sklearn.metrics import f1_score
from typing import Any, Dict, Optional
from datasets import Dataset
from transformers import AutoTokenizer
from omegaconf import ListConfig
from src.model.base_model import BaseModel
from src.utils.registry import MODEL_REGISTRY
from unsloth import FastLanguageModel


@MODEL_REGISTRY.register("unsloth")
class UnslothModel(BaseModel):
    """
    Unsloth를 사용한 최적화된 모델 (SFT + LoRA + Logit Selection).
    Unsloth는 모델 로딩과 학습 속도를 최적화합니다.
    """

    @staticmethod
    def get_tokenizer(model_name_or_path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        return tokenizer

    def __init__(self, model_name_or_path: str, use_peft: bool = True, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.use_peft = use_peft
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise Exception("cuda is not available")
        # max_seq_length는 학습 시 사용할 최대 시퀀스 길이
        self.max_seq_length = kwargs.get("max_seq_length", 2048)
        max_seq_length = self.max_seq_length

        # Unsloth FastLanguageModel로 모델 로드
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            load_in_4bit=kwargs.get("load_in_4bit", False),  # 4bit 양자화 옵션
            fast_inference=kwargs.get("fast_inference", True),
            max_lora_rank=kwargs.get("max_lora_rank", None),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", None),
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        if self.use_peft:
            # Hydra의 ListConfig를 일반 Python 리스트로 변환
            target_modules = kwargs.get("lora_target_modules", ["q_proj", "k_proj"])
            if isinstance(target_modules, ListConfig):
                target_modules = list(target_modules)

            # Unsloth의 최적화된 LoRA 설정
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=kwargs.get("lora_r", 6),
                target_modules=target_modules,
                lora_alpha=kwargs.get("lora_alpha", 8),
                lora_dropout=kwargs.get("lora_dropout", 0.05),
                bias=kwargs.get("lora_bias", "none"),
                use_gradient_checkpointing=kwargs.get("use_gradient_checkpointing", "unsloth"),
                random_state=kwargs.get("random_state", 3407),
                use_rslora=kwargs.get("use_rslora", False),  # Rank-Stabilized LoRA
                loftq_config=kwargs.get("loftq_config", None),  # LoftQ 양자화
            )
            self.model.print_trainable_parameters()

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, **kwargs):
        """
        TRL의 SFTTrainer를 사용한 학습 수행 (Logit Selection 전처리 포함)
        Unsloth 모델은 최적화된 학습 속도를 제공합니다.
        """

        # 모델의 logits를 조정하여 정답 토큰(1~5) 부분만 출력하도록 설정
        # def preprocess_logits_for_metrics(logits, labels):
        #     logits = logits if not isinstance(logits, tuple) else logits[0]
        #     # 숫자 1~5에 해당하는 토큰 ID 추출
        #     logit_idx = [self.tokenizer.vocab[str(i)] for i in range(1, 6)]
        #     # 마지막에서 두 번째 토큰(-2)이 생성된 정답 토큰임 (마지막은 EOS)
        #     logits = logits[:, -2, logit_idx]
        #     return logits

        # # Metric 계산 함수 (Macro F1)
        # def compute_metrics(evaluation_result):
        #     logits, labels = evaluation_result

        #     # 레이블 전처리 (-100 무시)
        #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        #     decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        #     # 정답 문자열에서 숫자 추출 (문자열 -> 정수 인덱스 0~4)
        #     label_indices = []
        #     for l in decoded_labels:
        #         clean_l = l.split("<end_of_turn>")[0].strip()
        #         label_indices.append(int(clean_l) - 1 if clean_l in "12345" else 0)

        #     # 예측값 계산 (Softmax 후 argmax)
        #     preds = np.argmax(logits, axis=-1)

        #     # Macro F1 계산
        #     f1 = f1_score(label_indices, preds, average="macro", zero_division=0)
        #     return {"macro_f1": f1}

        # data_collator = DataCollatorForCompletionOnlyLM(
        #     response_template="<start_of_turn>model",
        #     tokenizer=self.tokenizer,
        # )

        # # kwargs에 있는 설정들을 SFTConfig로 전달
        # training_args = SFTConfig(
        #     output_dir=kwargs.get("output_dir", "./output"),
        #     num_train_epochs=kwargs.get("num_train_epochs", 3),
        #     per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 1),
        #     per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 1),
        #     gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
        #     learning_rate=kwargs.get("learning_rate", 2e-5),
        #     lr_scheduler_type=kwargs.get("lr_scheduler_type", "cosine"),
        #     weight_decay=kwargs.get("weight_decay", 0.01),
        #     logging_steps=kwargs.get("logging_steps", 100),
        #     eval_steps=kwargs.get("eval_steps", 50),
        #     save_strategy=kwargs.get("save_strategy", "epoch"),
        #     eval_strategy=kwargs.get("evaluation_strategy", "epoch"),
        #     save_total_limit=kwargs.get("save_total_limit", 2),
        #     save_only_model=kwargs.get("save_only_model", True),
        #     fp16=kwargs.get("fp16", True),
        #     bf16=kwargs.get("bf16", False),
        #     report_to=kwargs.get("report_to", "none"),
        #     packing=False,
        # )

        # trainer = SFTTrainer(
        #     model=self.model,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     args=training_args,
        #     data_collator=data_collator,
        #     compute_metrics=compute_metrics,
        #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # )

        # trainer.train()

        # # 학습 끝난 후 저장
        # if kwargs.get("save_model", True):
        #     self.save_model(kwargs.get("output_dir", "./output"))
        pass

    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, float]:
        """
        모델 성능 평가 (Logit Selection + Macro F1)
        """
        self.model.eval()
        FastLanguageModel.for_inference(self.model)
        infer_results = []
        labels = []

        # 숫자 1~5에 해당하는 토큰 ID 준비
        target_token_ids = [self.tokenizer.vocab[str(i)] for i in range(1, 6)]

        for example in dataset:
            inputs = torch.tensor([example["input_ids"]]).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                # 마지막 토큰의 logits 추출 (-1)
                logits = outputs.logits[:, -1, target_token_ids].flatten().cpu()
                # Softmax 적용
                probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
                # 가장 확률 높은 인덱스(0~4) 선택
                pred_idx = np.argmax(probs)
                infer_results.append(pred_idx)

                # 정답 라벨 처리 (1~5 -> 0~4)
                if "answer" in example:
                    label_val = int(example["answer"]) - 1
                elif "label" in example:  # preprocess 결과에 따라 컬럼명이 다를 수 있음
                    label_val = int(example["label"]) - 1
                else:
                    label_val = 0  # Default
                labels.append(label_val)

        score = f1_score(labels, infer_results, average="macro", zero_division=0)
        return {"macro_f1": score}

    def predict(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        추론 수행 (Logit Selection 방식)
        """
        self.model.eval()
        FastLanguageModel.for_inference(self.model)
        predictions = {}

        target_token_ids = [self.tokenizer.vocab[str(i)] for i in range(1, 6)]
        pred_choices_map = {i: str(i + 1) for i in range(5)}  # 0->'1', 1->'2'...

        for example in dataset:
            inputs = torch.tensor([example["input_ids"]]).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[:, -1, target_token_ids].flatten().cpu()
                probs = torch.nn.functional.softmax(logits, dim=-1).numpy()

                # 가장 확률 높은 정답 선택
                pred_val = pred_choices_map[np.argmax(probs)]
                predictions[example["id"]] = pred_val

        return predictions

    def save_model(self, save_path: str):
        """
        Unsloth 모델 저장 (최적화된 형식)
        """
        # Unsloth의 최적화된 저장 메서드 사용
        # self.model.save_pretrained(save_path)
        # self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path: str):
        """
        저장된 Unsloth 모델 로드
        """
        # Unsloth 모델 로드
        # max_seq_length = getattr(self, "max_seq_length", 1024)  # __init__에서 설정된 값 사용
        # # dtype = None
        # # if is_bfloat16_supported():
        # #     dtype = "bfloat16"
        # # else:
        # #     dtype = "float16"

        # self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        #     model_name=load_path,
        #     max_seq_length=max_seq_length,
        #     dtype=dtype,
        #     load_in_4bit=False,  # 로드 시에는 4bit 양자화 비활성화
        #     trust_remote_code=True,
        # )

        # # 토크나이저 설정
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.padding_side = "right"
        # self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
