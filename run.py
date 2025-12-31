# from unsloth import FastLanguageModel  # unsloth import는 반드시 최상단에 위치해야함
import hydra
import pandas as pd
import os
import re
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# from transformers import AutoTokenizer
from hydra.core.hydra_config import HydraConfig

import src.model
import src.data
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
from src.utils import set_seed, get_logger, setup_logging, wait_for_gpu_availability

# 레지스트리에 모델과 데이터셋을 등록하기 위해 import
# __init__.py에서 자동으로 baseline_model과 baseline_data를 import함

logger = get_logger(__name__)


# TODO: 시작 - 끝 타이머 추가 / 종료 기록
# Hydra를 통해 설정 파일을 로드합니다.
# config_path는 프로젝트 루트 기준으로 설정
@hydra.main(version_base=None, config_path="config", config_name="test_config")
def main(cfg: DictConfig):
    # 로깅 설정 (config에서 읽어옴)
    setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        use_color=cfg.get("logging", {}).get("use_color", True),
    )

    # 난수 시드 고정
    wait_for_gpu_availability()
    set_seed(cfg.seed)
    hydra_cfg = HydraConfig.get()
    # logger.debug(f"설정 파일:\n{OmegaConf.to_yaml(cfg)}")
    logger.debug(f"mode(train, inference, evaluate): {cfg.mode}")
    current_config_name = hydra_cfg.job.config_name
    logger.debug(f"config file path: {current_config_name}")
    logger.debug(f"model_type: {cfg.model.type}")
    logger.debug(f"model_name_or_path: {cfg.model.model_name_or_path}")
    logger.debug(f"dataset_type: {cfg.dataset.type}")
    print(OmegaConf.to_yaml(cfg))

    # 모델 클래스 로드 및 Tokenizer 초기화
    # 모델에 맞는 토크나이저(Chat Template 포함)를 가져오기 위해 모델 클래스를 먼저 로드합니다.
    model_cls = MODEL_REGISTRY.get(cfg.model.type)

    # 실행 모드에 따른 동작 수행
    if cfg.mode == "train":
        logger.info("학습 모드 시작")
        logger.info(f"모델 인스턴스화: {model_cls.__name__}")
        # Model 초기화
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            use_peft=cfg.model.use_peft,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            lora_target_modules=cfg.model.lora_target_modules,
            max_seq_length=cfg.model.max_seq_length,
            lora_bias=cfg.model.lora_bias,
            **cfg.training,  # 학습 관련 설정 전달
        )
        tokenizer = model.tokenizer

        # Dataset 클래스 로드
        logger.info(f"데이터셋 인스턴스화: {cfg.dataset.type}")
        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)

        # Dataset 로드 및 전처리
        logger.info("데이터셋 전처리 시작")
        processed_dataset = dataset.preprocess(
            tokenizer,
            max_length=cfg.model.max_seq_length,
            template=cfg.prompt.name,
            **cfg.dataset.preprocess.train,
        )

        logger.info("데이터셋 전처리 완료")

        # 학습 데이터셋과 검증 데이터셋 분리 (임시로 9:1 분할)
        split_dataset = processed_dataset["train"].train_test_split(test_size=0.1, seed=cfg.seed)

        model.train(
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            **cfg.training,
        )

    elif cfg.mode == "inference":
        logger.info("추론 모드 시작")

        # TODO: 추후에 인터페이스 변경하지 않고 진행
        # model_cls에 아예 cfg나 .yaml 경로를 통째로 넘겨서 내부 로직에서 처리하도록 변경
        # 2-1. Model 초기화
        # Ollama 모델 여부 확인 (tokenizer가 없는 모델)
        is_ollama_model = cfg.model.type in ["qwen3-ollama"]
        logger.debug(f"is_ollama_model: {is_ollama_model}")

        if is_ollama_model:
            logger.debug("Ollama 모델로 인식됨")
            # Ollama 모델은 간단히 초기화 (ollama 설정 전달)

            try:
                ollama_config = OmegaConf.to_container(cfg.inference.get("ollama"), resolve=True)
            except Exception:
                raise ValueError("inference.ollama 설정을 불러오는 데 실패했습니다.")

            logger.debug(f"ollama_config: {ollama_config}")
            model = model_cls(
                model_name_or_path=cfg.model.get("model_name_or_path", ""),
                ollama=ollama_config,
            )
            tokenizer = None  # Ollama 모델은 토크나이저 불필요
        else:
            logger.debug("기존 HuggingFace 모델로 인식됨")
            # 기존 HuggingFace 모델 초기화
            model = model_cls(
                model_name_or_path=cfg.model.model_name_or_path,
                use_peft=cfg.model.use_peft,
                lora_r=cfg.model.lora_r,
                lora_alpha=cfg.model.lora_alpha,
                lora_dropout=cfg.model.lora_dropout,
                lora_target_modules=cfg.model.lora_target_modules,
                max_seq_length=cfg.model.max_seq_length,
                lora_bias=cfg.model.lora_bias,
            )
            tokenizer = model.tokenizer

        # 2-2. 학습된 모델 로드
        # model_load_path = cfg.inference.get("model_load_path", cfg.training.output_dir)
        # logger.debug(f"모델 로드 경로: {model_load_path}")
        # if not os.path.exists(model_load_path):
        #     raise ValueError(f"모델 경로를 찾을 수 없습니다: {model_load_path}")
        # # 체크포인트 디렉토리가 여러 개일 경우 가장 마지막 것을 로드
        # p = Path(model_load_path)
        # ckpts = [d for d in p.glob("checkpoint-*") if d.is_dir()]
        # if ckpts:
        #     ckpts.sort(
        #         key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1))
        #     )
        #     model_load_path = str(ckpts[-1])

        # print(f"모델 로드 중: {model_load_path}")
        # model.load_model(model_load_path)

        # 2-3. test.csv 로드 및 전처리
        test_dataset_path = cfg.inference.get("test_dataset_path", "data/test.csv")
        if not os.path.exists(test_dataset_path):
            raise ValueError(f"테스트 데이터셋 경로를 찾을 수 없습니다: {test_dataset_path}")

        logger.info(f"테스트 데이터셋 로드 중: {test_dataset_path}")
        test_dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        test_dataset = test_dataset_cls(test_dataset_path)
        logger.info("테스트 데이터셋 전처리 시작")
        # logger.info(f"전처리 옵션: {cfg.model.max_seq_length}, {cfg.prompt.name}, {cfg.dataset.preprocess.inference}")
        logger.info(f"cfg inference {cfg.inference}")
        processed_test_dataset = test_dataset.preprocess(
            tokenizer,
            max_length=cfg.model.max_seq_length,
            template=cfg.prompt.name,
            **cfg.dataset.preprocess.inference,
        )
        logger.info("테스트 데이터셋 전처리 완료")

        # 2-4. 추론 수행
        predictions = model.predict(dataset=processed_test_dataset["train"], **cfg.inference)

        # 2-5. 결과 출력 (일부만)
        logger.info(f"총 {len(predictions)}개 예측 완료")
        logger.info(f"예측 결과 샘플: {list(predictions.items())[:3]}")

        # 2-6. output.csv 저장
        # TODO: 경로 깔끔하게 정리
        # 파일명 덮어쓰지 않도록 수정
        output_path = Path(cfg.inference.get("output_path", "outputs/fallback"))
        os.makedirs(output_path, exist_ok=True)
        file_name = f"{current_config_name}_output.csv"
        final_output_path = output_path.joinpath(file_name)

        # predictions 딕셔너리를 DataFrame으로 변환
        df_output = pd.DataFrame([{"id": id, "answer": answer} for id, answer in predictions.items()])
        df_output = df_output.sort_values("id")  # id 순서대로 정렬
        logger.info(f"결과를 {final_output_path}에 저장합니다.")
        df_output.to_csv(final_output_path, index=False)
        print(f"결과 저장 완료: {final_output_path}")

    elif cfg.mode == "evaluate":
        print("🚀 평가 모드 시작")

        # Model 초기화
        model = model_cls(
            model_name_or_path=cfg.model.model_name_or_path,
            use_peft=cfg.model.use_peft,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            lora_target_modules=cfg.model.lora_target_modules,
            lora_bias=cfg.model.lora_bias,
            max_seq_length=cfg.model.max_seq_length,
        )

        tokenizer = model.tokenizer

        # Dataset 로드 및 전처리
        dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        dataset = dataset_cls(cfg.dataset.path)
        processed_dataset = dataset.preprocess(
            tokenizer,
            max_length=cfg.model.max_seq_length,
            template=cfg.prompt.name,
            **cfg.dataset.preprocess.inference,
        )

        # 2-3. 학습된 모델 로드 (선택사항)
        # model_load_path = cfg.inference.get("model_load_path", cfg.training.output_dir)
        # if os.path.exists(model_load_path):
        # pass
        # print(f"모델 로드 중: {model_load_path}")
        # model.load_model(model_load_path)

        # 2-4. 학습 데이터셋 일부를 사용하여 평가 (임시)
        split_dataset = processed_dataset["train"].train_test_split(test_size=0.1, seed=cfg.seed)
        metrics = model.evaluate(split_dataset["test"])
        print(f"평가 결과: {metrics}")


if __name__ == "__main__":
    main()
