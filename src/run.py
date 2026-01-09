import unsloth 
import hydra
import pandas as pd
import os
import sys
import re
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
from src.utils.utils import set_seed

import src.model  
import src.data  

# Hydra를 통해 설정 파일을 로드합니다.
@hydra.main(version_base=None, config_path=str(project_root / "config"), config_name="config")
def main(cfg: DictConfig):
    # 난수 시드 고정
    set_seed(cfg.seed)
    
    print(OmegaConf.to_yaml(cfg))

    # 모델 클래스 로드 및 Tokenizer 초기화
    model_cls = MODEL_REGISTRY.get(cfg.model.type)
    
    if cfg.mode == "train":
        print("🚀 학습 모드 시작")
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            use_peft=cfg.model.use_peft,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            lora_target_modules=cfg.model.lora_target_modules,
            max_seq_length=cfg.model.max_seq_length,
            lora_bias=cfg.model.lora_bias,
            **cfg.training 
        )

        tokenizer = model.tokenizer

        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)

        # Dataset 로드 및 전처리
        processed_dataset = dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **cfg.dataset.preprocess.train
        )
        from datasets import DatasetDict
        
        full_dataset = processed_dataset["train"]
        # dataset split
        split_ratio = cfg.dataset.get("split_ratio", 0.8)  

        split_dataset = full_dataset.train_test_split(
            train_size=split_ratio,
            seed=cfg.seed
        )
        
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        model.train(
            train_dataset=train_dataset,
            eval_dataset=None,
            **cfg.training
        )
        
    elif cfg.mode == "inference":
        print("🚀 추론 모드 시작")
        
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            skip_init=True,  
            max_seq_length=cfg.model.max_seq_length,
        )
        tokenizer = model.tokenizer
        
        model_load_path = cfg.inference.get("model_load_path", cfg.training.output_dir)
        if not os.path.exists(model_load_path):
            raise ValueError(f"모델 경로를 찾을 수 없습니다: {model_load_path}")
        # 체크포인트 디렉토리가 여러 개일 경우 가장 마지막 것을 로드
        p = Path(model_load_path)
        ckpts = [d for d in p.glob("checkpoint-*") if d.is_dir()]
        if ckpts:
            ckpts.sort(
                key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1))
            )
            model_load_path = str(ckpts[-1])

        print(f"모델 로드 중: {model_load_path}")
        model.load_model(model_load_path)
        
        test_dataset_path = cfg.inference.get("test_dataset_path", "data/test.csv")
        
        print(f"테스트 데이터셋 로드 중: {test_dataset_path}")
        test_dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        test_dataset = test_dataset_cls(test_dataset_path)
        processed_test_dataset = test_dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **cfg.dataset.preprocess.inference
        )

        predictions = model.predict(
            dataset=processed_test_dataset["train"],
            **cfg.inference
        )
        

        output_path = cfg.inference.get("output_path", "output/output.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_output = pd.DataFrame([
            {"id": id, "answer": answer} 
            for id, answer in predictions.items()
        ])
        df_output = df_output.sort_values("id")
        df_output.to_csv(output_path, index=False)
        print(f"결과 저장 완료: {output_path}")
        
    elif cfg.mode == "evaluate":
        print("🚀 평가 모드 시작")

        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            skip_init=True, 
            max_seq_length=cfg.model.max_seq_length,
        )

        tokenizer = model.tokenizer

        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)

        eval_preprocess_config = cfg.dataset.preprocess.get("evaluate", cfg.dataset.preprocess.inference)

        processed_dataset = dataset.preprocess(
            tokenizer, 
            max_length=cfg.model.max_seq_length, 
            template=cfg.prompt.name, 
            **eval_preprocess_config  
        )
        # dataset split
        full_dataset = processed_dataset["train"]
        split_ratio = cfg.dataset.get("split_ratio", 0.8)
        

        split_dataset = full_dataset.train_test_split(
            train_size=split_ratio,
            seed=cfg.seed
        )
        

        eval_dataset = split_dataset["test"]

        model_load_path = cfg.evaluate.get("model_load_path", cfg.training.output_dir)

        p = Path(model_load_path)
        ckpts = [d for d in p.glob("checkpoint-*") if d.is_dir()]
        if ckpts:
            ckpts.sort(
                key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1))
            )
            model_load_path = str(ckpts[-1])
        
        print(f"모델 로드 중: {model_load_path}")
        model.load_model(model_load_path)
        
        metrics = model.evaluate(
            eval_dataset,
            original_dataset_path=cfg.dataset.path, 
            eval_output_path=cfg.evaluate.get("eval_output_path", "output/eval_results.csv")
        )
        print(f"평가 결과 : {metrics}")


if __name__ == "__main__":
    main()
