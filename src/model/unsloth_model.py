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
    
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = kwargs.get("max_seq_length", 2048)
        self.load_in_4bit = kwargs.get("load_in_4bit", True)

        # ✅ Unsloth에서 model + tokenizer를 동시에 로드
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ✅ LoRA
        if kwargs.get("use_peft", True):
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=kwargs.get("lora_r", 16),
                lora_alpha=kwargs.get("lora_alpha", 16),
                lora_dropout=kwargs.get("lora_dropout", 0),
                target_modules=kwargs.get(
                    "lora_target_modules",
                    [
                        "q_proj", "k_proj", "v_proj",
                        "o_proj", "gate_proj", "up_proj", "down_proj",
                    ],
                ),
                bias=kwargs.get("lora_bias", "none"),
                use_gradient_checkpointing="unsloth",
            )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        FastLanguageModel.for_training(self.model)

        model_name = self.model_name_or_path.lower()

        # ✅ 모델별 loss 전략 자동 분기
        if "gemma" in model_name:
            loss_config = dict(completion_only_loss=True)
        else:
            # llama / qwen
            loss_config = dict(assistant_only_loss=True)

        training_args = SFTConfig(
            output_dir=kwargs.get("output_dir", "./output"),
            num_train_epochs=kwargs.get("num_train_epochs", 3),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            learning_rate=kwargs.get("learning_rate", 2e-4),
            logging_steps=kwargs.get("logging_steps", 50),
            save_strategy=kwargs.get("save_strategy", "epoch"),
            eval_strategy=kwargs.get("eval_strategy", "epoch"),
            fp16=kwargs.get("fp16", True),
            bf16=kwargs.get("bf16", False),
            packing=False,
            report_to="none",
            **loss_config,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        # ================================
        # TRL entropy 계산 완전 차단 패치
        # ================================
        import types

        _original_compute_loss = trainer.compute_loss

        def compute_loss_no_entropy(self, model, inputs, return_outputs=False, **kwargs):
            # 원래 loss 계산
            outputs = model(**inputs)
            loss = outputs.loss

            if return_outputs:
                return loss, outputs
            return loss

        trainer.compute_loss = types.MethodType(compute_loss_no_entropy, trainer)


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

        for example in dataset:
            inputs = torch.tensor([example["input_ids"]]).to(self.device)
            with torch.no_grad():
                logits = self.model(inputs).logits
                scores = logits[:, -1, target_token_ids].cpu().numpy()
                pred = int(np.argmax(scores)) + 1
                predictions[example["id"]] = str(pred)

        return predictions

    def save_model(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path: str):
        """
        Base 모델을 다시 로드한 뒤
        LoRA adapter를 결합
        """
        # 1️⃣ Base 모델 재로딩
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name_or_path,   # ⭐ 원래 base 모델
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )

        # 2️⃣ LoRA adapter 로드
        self.model.load_adapter(load_path)
        # 3️⃣ 추론 모드로 전환
        FastLanguageModel.for_inference(self.model)

