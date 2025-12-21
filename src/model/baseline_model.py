import torch
import re
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
    대회 베이스라인 모델 (SFT + LoRA).
    AutoModelForCausalLM을 사용하여 Generation Task를 수행합니다.
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
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
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
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, **kwargs):
        """
        TRL의 SFTTrainer를 사용한 학습 수행
        """
        training_args = SFTConfig(
            output_dir=kwargs.get("output_dir", "./output"),
            num_train_epochs=kwargs.get("num_train_epochs", 1),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 4),
            learning_rate=kwargs.get("learning_rate", 2e-4),
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            # dataset_text_field="text", # 만약 전처리된 텍스트 컬럼을 쓸 경우 지정
            max_seq_length=kwargs.get("max_seq_length", 1024),
            packing=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        trainer.train()
        
        # 학습 끝난 후 저장
        if kwargs.get("save_model", True):
            self.save_model(kwargs.get("output_dir", "./output"))

    def evaluate(self, dataset: Dataset, **kwargs) -> Dict[str, float]:
        """
        모델 성능 평가 (Macro F1-score)
        """

        self.model.eval()
        preds = []
        labels = []

        # 배치 처리 없이 순차적으로 진행 (필요 시 배치 처리로 최적화 가능)
        for example in dataset:
            # 입력 데이터 준비
            if "input_ids" in example:
                inputs = torch.tensor([example["input_ids"]]).to(self.device)
            else:
                # 텍스트로 들어온 경우 (예외 처리)
                # preprocess 과정에서 'text' 컬럼을 만들었다면 그것을 사용
                text = example.get("text", "") 
                if not text:
                    # text 컬럼도 없으면 직접 구성
                    q = example.get('question', '')
                    p = example.get('paragraph', '')
                    text = f"지문: {p}\n질문: {q}"
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            # 정답 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 5), # 정답은 짧으므로
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 생성된 텍스트에서 정답 추출 (간단한 파싱)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 숫자(1~5) 추출 시도
            match = re.search(r'\b([1-5])\b', generated_text)
            pred_val = match.group(1) if match else "0" # 못 찾으면 0
            
            # 실제 정답 (Label)
            # preprocess에서 labels가 토큰화되어 있다면 디코딩 필요, 아니면 raw 데이터의 'answer' 사용
            if "answer" in example:
                label_val = str(example["answer"])
            elif "labels" in example:
                # labels 토큰을 디코딩
                label_ids = [l for l in example["labels"] if l != -100] # ignore_index 제외
                label_val = self.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
            else:
                label_val = "0"

            preds.append(pred_val)
            labels.append(label_val)

        # Macro F1 계산
        score = f1_score(labels, preds, average='macro', zero_division=0)
        return {"macro_f1": score}

    def predict(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        추론 수행: 질문을 주고 답변을 생성
        """
        self.model.eval()
        predictions = {}
        
        # 배치 단위로 추론하면 더 빠르지만, 예시를 위해 순차 처리
        for i, example in enumerate(dataset):
            # 전처리된 input_ids가 있다면 사용, 없다면 텍스트로 처리
            if "input_ids" in example:
                inputs = torch.tensor([example["input_ids"]]).to(self.device)
            else:
                # 텍스트가 들어온 경우 (예외 처리)
                text = f"지문: {example['paragraph']}\n질문: {example['question']}"
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 10),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 생성된 텍스트 디코딩
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions[example['id'] if 'id' in example else i] = generated_text

        return predictions

    def save_model(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path: str):
        # 로드 로직은 필요시 구현 (AutoModel.from_pretrained로 대체 가능)
        pass
