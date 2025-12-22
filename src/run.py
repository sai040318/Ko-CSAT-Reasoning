import hydra
import pandas as pd
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
from src.utils.utils import set_seed

# 레지스트리에 모델과 데이터셋을 등록하기 위해 import
# __init__.py에서 자동으로 baseline_model과 baseline_data를 import함
import src.model  # noqa: F401
import src.data  # noqa: F401 

# Hydra를 통해 설정 파일을 로드합니다.
# config_path는 프로젝트 루트 기준으로 설정
@hydra.main(version_base=None, config_path=str(project_root / "config"), config_name="config")
def main(cfg: DictConfig):
    # 난수 시드 고정
    set_seed(cfg.seed)
    
    print(OmegaConf.to_yaml(cfg))

    # 1. Tokenizer 로드 (데이터 전처리를 위해 필요)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
<<<<<<< HEAD
    # Gemma Chat Template 설정 (Baseline과 동일하게)
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
    
    # 2. 실행 모드에 따른 동작 수행
=======
    # 2. Dataset 로드 및 전처리
    dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
    dataset = dataset_cls(cfg.dataset.path)
    #processed_dataset = dataset.preprocess(tokenizer, max_length=cfg.model.max_seq_length)
    processed_dataset = dataset.preprocess(tokenizer, max_length=cfg.model.max_seq_length, template=cfg.prompt.name)

    # 3. Model 초기화
    model_cls = MODEL_REGISTRY.get(cfg.model.type)
    model = model_cls(
        model_name_or_path=cfg.model.model_name_or_path,
        use_peft=cfg.model.use_peft,
        **cfg.training # 학습 관련 설정 전달
    )

    # 4. 실행 모드에 따른 동작 수행
>>>>>>> 8eaa318fa1d4f06ab89ffe41adbcf99a13b59a9c
    if cfg.mode == "train":
        # 2-1. Dataset 로드 및 전처리
        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)
        processed_dataset = dataset.preprocess(tokenizer, max_length=cfg.model.max_seq_length)

        # 2-2. Model 초기화
        model_cls = MODEL_REGISTRY.get(cfg.model.type)
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            use_peft=cfg.model.use_peft,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            lora_target_modules=cfg.model.lora_target_modules,
            lora_bias=cfg.model.lora_bias,
            **cfg.training # 학습 관련 설정 전달
        )
        print("🚀 학습 모드 시작")
        # 학습 데이터셋과 검증 데이터셋 분리 (임시로 9:1 분할)
        split_dataset = processed_dataset["train"].train_test_split(test_size=0.1, seed=cfg.seed)
        
        model.train(
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            **cfg.training
        )
        
    elif cfg.mode == "inference":
        print("🚀 추론 모드 시작")
        
        # 2-1. Model 초기화 (구조만 생성, 가중치는 로드하지 않음)
        model_cls = MODEL_REGISTRY.get(cfg.model.type)
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            use_peft=cfg.model.use_peft,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            lora_target_modules=cfg.model.lora_target_modules,
            lora_bias=cfg.model.lora_bias,
        )
        
        # 2-2. 학습된 모델 로드
        model_load_path = cfg.inference.get("model_load_path", cfg.training.output_dir)
        if not os.path.exists(model_load_path):
            raise ValueError(f"모델 경로를 찾을 수 없습니다: {model_load_path}")
        print(f"모델 로드 중: {model_load_path}")
        model.load_model(model_load_path)
        
        # 2-3. test.csv 로드 및 전처리
        test_dataset_path = cfg.inference.get("test_dataset_path", "data/test.csv")
        if not os.path.exists(test_dataset_path):
            raise ValueError(f"테스트 데이터셋 경로를 찾을 수 없습니다: {test_dataset_path}")
        
        print(f"테스트 데이터셋 로드 중: {test_dataset_path}")
        test_dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        test_dataset = test_dataset_cls(test_dataset_path)
        processed_test_dataset = test_dataset.preprocess(tokenizer, max_length=cfg.model.max_seq_length)
        
        # 2-4. 추론 수행
        predictions = model.predict(
            dataset=processed_test_dataset["train"],
            **cfg.inference
        )
        
        # 2-5. 결과 출력 (일부만)
        print(f"총 {len(predictions)}개 예측 완료")
        print(f"샘플 예측 결과: {list(predictions.items())[:3]}")
        
        # 2-6. output.csv 저장
        output_path = cfg.inference.get("output_path", "output/output.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # predictions 딕셔너리를 DataFrame으로 변환
        df_output = pd.DataFrame([
            {"id": id, "answer": answer} 
            for id, answer in predictions.items()
        ])
        df_output = df_output.sort_values("id")  # id 순서대로 정렬
        df_output.to_csv(output_path, index=False)
        print(f"결과 저장 완료: {output_path}")
        
    elif cfg.mode == "evaluate":
        print("🚀 평가 모드 시작")
        
        # 2-1. Dataset 로드 및 전처리
        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)
        processed_dataset = dataset.preprocess(tokenizer, max_length=cfg.model.max_seq_length)

        # 2-2. Model 초기화
        model_cls = MODEL_REGISTRY.get(cfg.model.type)
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            use_peft=cfg.model.use_peft,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            lora_target_modules=cfg.model.lora_target_modules,
            lora_bias=cfg.model.lora_bias,
        )
        
        # 2-3. 학습된 모델 로드 (선택사항)
        model_load_path = cfg.inference.get("model_load_path", cfg.training.output_dir)
        if os.path.exists(model_load_path):
            print(f"모델 로드 중: {model_load_path}")
            model.load_model(model_load_path)
        
        # 2-4. 학습 데이터셋 일부를 사용하여 평가 (임시)
        split_dataset = processed_dataset["train"].train_test_split(test_size=0.1, seed=cfg.seed)
        metrics = model.evaluate(split_dataset["test"])
        print(f"평가 결과: {metrics}")

if __name__ == "__main__":
    main()
