import torch
import numpy as np
from sklearn.metrics import f1_score
from typing import Any, Dict, Optional
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from omegaconf import ListConfig
from unsloth import FastLanguageModel

from src.model.base_model import BaseModel
from src.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("unsloth")
class UnslothModel(BaseModel):
    """
    Unsloth 라이브러리를 활용한 가속화된 모델 클래스.
    """

    @staticmethod
    def get_tokenizer(model_name_or_path: str, **kwargs):
        """
        Unsloth 모델에 맞는 토크나이저 로드 및 Chat Template 설정.
        BaseModel의 기본 get_tokenizer를 오버라이드합니다.
        """
        from transformers import AutoTokenizer
        
        # 1. 기본 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        
        # 2. 시스템 프롬프트 지원을 위한 Chat Template 강제 설정 (Gemma 등 대응)
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
        
        return tokenizer
    
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Unsloth 설정
        self.max_seq_length = kwargs.get("max_seq_length", 2048)
        self.dtype = None  # None으로 설정하면 자동 감지 (Float16 or Bfloat16)
        self.load_in_4bit = kwargs.get("load_in_4bit", True)
        
        # 모델과 토크나이저 동시 로드
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name_or_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        
        # PEFT 설정
        if kwargs.get("use_peft", True):
            target_modules = kwargs.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            if isinstance(target_modules, ListConfig):
                target_modules = list(target_modules)
                
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=kwargs.get("lora_r", 16),
                target_modules=target_modules,
                lora_alpha=kwargs.get("lora_alpha", 16),
                lora_dropout=kwargs.get("lora_dropout", 0), # Unsloth는 dropout 0 권장
                bias=kwargs.get("lora_bias", "none"),
                use_gradient_checkpointing="unsloth", # Unsloth 최적화
                random_state=kwargs.get("seed", 3407),
                use_rslora=False,
                loftq_config=None,
            )

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, **kwargs):
        """
        Unsloth 최적화를 적용하여 학습 수행
        """
        # Unsloth 학습 모드 설정
        FastLanguageModel.for_training(self.model)

        def preprocess_logits_for_metrics(logits, labels):
            logits = logits if not isinstance(logits, tuple) else logits[0]
            logit_idx = [self.tokenizer.vocab[str(i)] for i in range(1, 6)]
            logits = logits[:, -2, logit_idx]
            return logits

        def compute_metrics(evaluation_result):
            logits, labels = evaluation_result
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            label_indices = []
            for l in decoded_labels:
                clean_l = l.split("<end_of_turn>")[0].strip()
                label_indices.append(int(clean_l) - 1 if clean_l in "12345" else 0)

            preds = np.argmax(logits, axis=-1)
            f1 = f1_score(label_indices, preds, average='macro', zero_division=0)
            return {"macro_f1": f1}

        # DataCollator 대신 SFTConfig의 최신 파라미터 사용
        training_args = SFTConfig(
            output_dir=kwargs.get("output_dir", "./output"),
            num_train_epochs=kwargs.get("num_train_epochs", 3),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            learning_rate=kwargs.get("learning_rate", 2e-4),
            weight_decay=kwargs.get("weight_decay", 0.01),
            lr_scheduler_type=kwargs.get("lr_scheduler_type", "linear"),
            seed=kwargs.get("seed", 3407),
            logging_steps=kwargs.get("logging_steps", 100),
            eval_steps=kwargs.get("eval_steps", 50),
            save_strategy=kwargs.get("save_strategy", "epoch"),
            eval_strategy=kwargs.get("evaluation_strategy", "epoch"),
            fp16=kwargs.get("fp16", not torch.cuda.is_bf16_supported()),
            bf16=kwargs.get("bf16", torch.cuda.is_bf16_supported()),
            packing=False,
            report_to=kwargs.get("report_to", "none"),
            completion_only_loss=True,
            response_template="<start_of_turn>model",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            args=training_args,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        trainer.train()
        
        if kwargs.get("save_model", True):
            self.save_model(kwargs.get("output_dir", "./output"))

    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, float]:
        # Unsloth 추론 모드 (속도 2배 향상)
        FastLanguageModel.for_inference(self.model)
        
        infer_results = []
        labels = []
        target_token_ids = [self.tokenizer.vocab[str(i)] for i in range(1, 6)]

        for example in dataset:
            inputs = torch.tensor([example["input_ids"]]).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[:, -1, target_token_ids].flatten().cpu()
                probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
                pred_idx = np.argmax(probs)
                infer_results.append(pred_idx)
                
                if "answer" in example:
                    label_val = int(example["answer"]) - 1
                elif "label" in example:
                    label_val = int(example["label"]) - 1
                else:
                    label_val = 0
                labels.append(label_val)

        score = f1_score(labels, infer_results, average='macro', zero_division=0)
        return {"macro_f1": score}

    def predict(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        FastLanguageModel.for_inference(self.model)
        predictions = {}
        target_token_ids = [self.tokenizer.vocab[str(i)] for i in range(1, 6)]
        pred_choices_map = {i: str(i+1) for i in range(5)}

        for example in dataset:
            inputs = torch.tensor([example["input_ids"]]).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[:, -1, target_token_ids].flatten().cpu()
                probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
                pred_val = pred_choices_map[np.argmax(probs)]
                predictions[example['id']] = pred_val

        return predictions

    def save_model(self, save_path: str):
        # Unsloth 전용 저장 방식 (GGUF 변환 등도 가능하나 기본 LoRA 저장)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path: str):
        # 학습된 어댑터 로드
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_path, # 저장된 경로
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
