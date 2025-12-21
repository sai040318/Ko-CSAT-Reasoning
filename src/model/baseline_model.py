import torch
import numpy as np
from sklearn.metrics import f1_score
from typing import Any, Dict, Optional, List
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from src.model.base_model import BaseModel
from src.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("baseline")
class BaselineModel(BaseModel):
    """
    대회 베이스라인 모델 (SFT + LoRA + Logit Selection).
    """

    def __init__(self, model_name_or_path: str, use_peft: bool = True, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.use_peft = use_peft
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 및 토크나이저 로드 (초기화)
        # 양자화 설정 (메모리 효율을 위해)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=bnb_config if self.use_peft else None,
            torch_dtype=torch.float16, # Baseline은 float16 사용
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        
        # Gemma Chat Template 설정 (Baseline과 동일하게)
        self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"

        if self.use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=kwargs.get("lora_r", 6),
                lora_alpha=kwargs.get("lora_alpha", 8),
                lora_dropout=kwargs.get("lora_dropout", 0.05),
                target_modules=kwargs.get("lora_target_modules", ['q_proj', 'k_proj']),
                bias=kwargs.get("lora_bias", "none"),
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, **kwargs):
        """
        TRL의 SFTTrainer를 사용한 학습 수행 (Logit Selection 전처리 포함)
        """
        # 모델의 logits를 조정하여 정답 토큰(1~5) 부분만 출력하도록 설정
        def preprocess_logits_for_metrics(logits, labels):
            logits = logits if not isinstance(logits, tuple) else logits[0]
            # 숫자 1~5에 해당하는 토큰 ID 추출
            logit_idx = [self.tokenizer.vocab[str(i)] for i in range(1, 6)]
            # 마지막에서 두 번째 토큰(-2)이 생성된 정답 토큰임 (마지막은 EOS)
            logits = logits[:, -2, logit_idx]
            return logits

        # Metric 계산 함수 (Macro F1)
        def compute_metrics(evaluation_result):
            logits, labels = evaluation_result
            
            # 레이블 전처리 (-100 무시)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # 정답 문자열에서 숫자 추출 (문자열 -> 정수 인덱스 0~4)
            label_indices = []
            for l in decoded_labels:
                clean_l = l.split("<end_of_turn>")[0].strip()
                label_indices.append(int(clean_l) - 1 if clean_l in "12345" else 0)

            # 예측값 계산 (Softmax 후 argmax)
            preds = np.argmax(logits, axis=-1)
            
            # Macro F1 계산
            f1 = f1_score(label_indices, preds, average='macro', zero_division=0)
            return {"macro_f1": f1}

        # kwargs에 있는 설정들을 SFTConfig로 전달
        training_args = SFTConfig(
            output_dir=kwargs.get("output_dir", "./output"),
            num_train_epochs=kwargs.get("num_train_epochs", 3),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            learning_rate=kwargs.get("learning_rate", 2e-5),
            lr_scheduler_type=kwargs.get("lr_scheduler_type", "cosine"),
            weight_decay=kwargs.get("weight_decay", 0.01),
            logging_steps=kwargs.get("logging_steps", 1),
            eval_steps=kwargs.get("eval_steps", 50),
            save_strategy=kwargs.get("save_strategy", "epoch"),
            evaluation_strategy=kwargs.get("evaluation_strategy", "epoch"),
            save_total_limit=kwargs.get("save_total_limit", 2),
            save_only_model=kwargs.get("save_only_model", True),
            fp16=kwargs.get("fp16", True),
            report_to=kwargs.get("report_to", "none"),
            max_seq_length=kwargs.get("max_seq_length", 1024),
            packing=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        trainer.train()
        
        # 학습 끝난 후 저장
        if kwargs.get("save_model", True):
            self.save_model(kwargs.get("output_dir", "./output"))

    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, float]:
        """
        모델 성능 평가 (Logit Selection + Macro F1)
        """
        self.model.eval()
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
                elif "label" in example: # preprocess 결과에 따라 컬럼명이 다를 수 있음
                    label_val = int(example["label"]) - 1
                else:
                    label_val = 0 # Default
                labels.append(label_val)

        score = f1_score(labels, infer_results, average='macro', zero_division=0)
        return {"macro_f1": score}

    def predict(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        추론 수행 (Logit Selection 방식)
        """
        self.model.eval()
        predictions = {}
        
        target_token_ids = [self.tokenizer.vocab[str(i)] for i in range(1, 6)]
        pred_choices_map = {i: str(i+1) for i in range(5)} # 0->'1', 1->'2'...

        for example in dataset:
            inputs = torch.tensor([example["input_ids"]]).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[:, -1, target_token_ids].flatten().cpu()
                probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
                
                # 가장 확률 높은 정답 선택
                pred_val = pred_choices_map[np.argmax(probs)]
                predictions[example['id']] = pred_val

        return predictions

    def save_model(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path: str):
        # 로드 로직은 필요시 구현 (AutoModel.from_pretrained로 대체 가능)
        pass