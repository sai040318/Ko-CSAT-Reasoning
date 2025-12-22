import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
from src.utils.utils import set_seed

# Hydra를 통해 설정 파일을 로드합니다.
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # 난수 시드 고정
    set_seed(cfg.seed)
    
    print(OmegaConf.to_yaml(cfg))

    # 1. Tokenizer 로드 (데이터 전처리를 위해 필요)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    
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
    if cfg.mode == "train":
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
        # 추론 수행 (테스트 데이터셋 전체에 대해)
        predictions = model.predict(
            dataset=processed_dataset["train"], # 테스트 셋이라면 'test' 키 사용
            **cfg.inference
        )
        
        # 결과 출력 (일부만)
        print(f"총 {len(predictions)}개 예측 완료")
        print(f"샘플 예측 결과: {list(predictions.items())[:3]}")
        
        # TODO: submission.csv 저장 로직 추가
        
    elif cfg.mode == "evaluate":
        print("🚀 평가 모드 시작")
        # 학습 데이터셋 일부를 사용하여 평가 (임시)
        split_dataset = processed_dataset["train"].train_test_split(test_size=0.1, seed=cfg.seed)
        metrics = model.evaluate(split_dataset["test"])
        print(f"평가 결과: {metrics}")

if __name__ == "__main__":
    main()
