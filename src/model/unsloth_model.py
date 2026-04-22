import unsloth 
import torch
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import f1_score
from typing import Any, Dict, Optional
from datasets import Dataset
from omegaconf import ListConfig
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

from src.model.base_model import BaseModel
from src.utils.registry import MODEL_REGISTRY
# 레지스트리 등록
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

        skip_init = kwargs.get("skip_init", False)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            attn_implementation="sdpa",  
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if not skip_init and kwargs.get("use_peft", True):
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

        loss_config = dict(completion_only_loss=True)


        training_args = SFTConfig(
            output_dir=kwargs.get("output_dir", "./output"),
            num_train_epochs=kwargs.get("num_train_epochs", 3),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            learning_rate=kwargs.get("learning_rate", 2e-4),
            lr_scheduler_type=kwargs.get("lr_scheduler_type", "linear"),
            warmup_ratio=kwargs.get("warmup_ratio", 0.0),
            weight_decay=kwargs.get("weight_decay", 0.0),
            max_grad_norm=kwargs.get("max_grad_norm", 1.0),
            logging_steps=kwargs.get("logging_steps", 50),
            save_strategy=kwargs.get("save_strategy", "epoch"),
            eval_strategy=kwargs.get("eval_strategy", "epoch"),
            save_total_limit=kwargs.get("save_total_limit", None),
            fp16=kwargs.get("fp16", True),
            bf16=kwargs.get("bf16", False),
            packing=kwargs.get("packing", True),
            report_to=kwargs.get("report_to", "none"),
            **loss_config,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=None if training_args.eval_strategy == "no" else eval_dataset,
            args=training_args,
            processing_class=self.tokenizer,
        )

        import types

        _original_compute_loss = trainer.compute_loss

        def compute_loss_no_entropy(self, model, inputs, return_outputs=False, **kwargs):
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
        """
        평가 데이터셋에 대해 Macro F1 score를 계산하고 결과 CSV를 생성합니다.
        """
        FastLanguageModel.for_inference(self.model)
        
        infer_results = []
        generated_texts = []
        labels = []
        ids = []
        
        correct = 0
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
            generated_text = self._generate_choice_answer(example, max_new_tokens=kwargs.get("max_new_tokens", 8))
            pred_idx = self._extract_choice_index(generated_text)

            infer_results.append(pred_idx)
            generated_texts.append(generated_text)

            label_val = int(example["answer"]) - 1 if example["answer"] else 0
            labels.append(label_val)

            if "id" in example:
                ids.append(example["id"])
            else:
                ids.append(idx)
            
            # 정답 체크
            if pred_idx == label_val:
                correct += 1
        
        # 메트릭 계산
        accuracy = correct / len(dataset)
        macro_f1 = f1_score(labels, infer_results, average='macro', zero_division=0)
        
        print(f"\n{'='*60}")
        print(f"📈 평가 결과")
        print(f"  - Accuracy: {accuracy:.4f} ({correct}/{len(dataset)})")
        print(f"  - Macro F1: {macro_f1:.4f}")
        
        # 결과 CSV 저장
        # 에러케이스 분석용
        eval_output_path = kwargs.get("eval_output_path", None)
        if eval_output_path:
            eval_dataset_path = kwargs.get(
                "eval_dataset_path",
                kwargs.get("original_dataset_path", None),
            )
            if eval_dataset_path and os.path.exists(eval_dataset_path):
                df_original = pd.read_csv(eval_dataset_path)

                df_original["predict"] = [infer_results[i] + 1 for i in range(len(infer_results))]  # 1~5로 변환
                df_original["correct"] = ["O" if infer_results[i] == labels[i] else "X" for i in range(len(infer_results))]
                df_original["generated_text"] = generated_texts

                os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
                df_original.to_csv(eval_output_path, index=False)
                print(f"✅ 평가 결과 CSV 저장 완료: {eval_output_path}")
        
        return {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(dataset)
        }

    def predict(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        FastLanguageModel.for_inference(self.model)

        predictions = {}

        for example in tqdm(dataset, desc="Predicting", total=len(dataset)):
            generated_text = self._generate_choice_answer(example, max_new_tokens=kwargs.get("max_new_tokens", 8))
            pred = self._extract_choice_index(generated_text) + 1
            predictions[example["id"]] = str(pred)

        return predictions

    def save_model(self, save_path: str):
        import json
        from pathlib import Path as _Path
        _Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # Unsloth 버전에 따라 adapter_config.json이 저장되지 않을 수 있으므로 직접 보장
        adapter_config_path = _Path(save_path) / "adapter_config.json"
        if not adapter_config_path.exists() and hasattr(self.model, "peft_config"):
            config = list(self.model.peft_config.values())[0]
            adapter_config_path.write_text(json.dumps(config.to_dict(), indent=2))

    def load_model(self, load_path: str):
        """
        저장된 LoRA adapter를 로드
        """
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            self.model, 
            load_path,
            is_trainable=False 
        )

        FastLanguageModel.for_inference(self.model)

    def _generate_choice_answer(self, example: Dict[str, Any], max_new_tokens: int = 8) -> str:
        input_ids = torch.tensor([example["input_ids"]], device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0, input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_choice_index(text: str) -> int:
        stripped = text.strip()

        # 생성 직후 첫 토큰만 정답으로 간주한다.
        leading_digit = re.match(r"^([1-5])(?:\s|[-:.)]|$)", stripped)
        if leading_digit:
            return int(leading_digit.group(1)) - 1

        leading_circled = re.match(r"^([①②③④⑤])", stripped)
        if leading_circled:
            return "①②③④⑤".index(leading_circled.group(1))

        first_line = stripped.splitlines()[0].strip() if stripped else ""

        line_digit = re.match(r"^([1-5])(?:\s|[-:.)]|$)", first_line)
        if line_digit:
            return int(line_digit.group(1)) - 1

        line_circled = re.match(r"^([①②③④⑤])", first_line)
        if line_circled:
            return "①②③④⑤".index(line_circled.group(1))

        return 0
