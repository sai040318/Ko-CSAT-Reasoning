import hydra
import pandas as pd
import os
import re
import time
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.utils.registry import MODEL_REGISTRY, DATASET_REGISTRY
from src.utils import (
    set_seed,
    get_logger,
    setup_logging,
    wait_for_gpu_availability,
    log_gpu_status,
    ExperimentLogger,
    tune_third_party_log_levels,
)
from hydra.core.hydra_config import HydraConfig


logger = None


# Hydra를 통해 설정 파일을 로드합니다.
# config_path는 프로젝트 루트 기준으로 설정
@hydra.main(version_base=None, config_path="config", config_name="qwen3_2507_thinking")
def main(cfg: DictConfig):
    global logger
    # 타이머 시작
    start_time = time.time()

    # 로깅 설정 (config에서 읽어옴)
    setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        use_color=cfg.get("logging", {}).get("use_color", True),
    )
    logger = get_logger(__name__)

    import src.model
    import src.data

    if cfg.get("logging", {}).get("tune_third_party", False):
        tune_third_party_log_levels()
    exp_logger = ExperimentLogger(exp_name=cfg.get("exp_name"))

    # 난수 시드 고정
    set_seed(cfg.seed)
    hydra_cfg = HydraConfig.get()
    # logger.debug(f"설정 파일:\n{OmegaConf.to_yaml(cfg)}")
    logger.debug(f"mode(train, inference, evaluate): {cfg.mode}")
    current_config_name = hydra_cfg.job.config_name
    logger.debug(f"config file path: {current_config_name}")
    logger.debug(f"model_type: {cfg.model.type}")
    logger.debug(f"model_name_or_path: {cfg.model.model_name_or_path}")
    logger.debug(f"dataset_type: {cfg.dataset.type}")
    logger.debug(f"config file content:\n{OmegaConf.to_yaml(cfg)}")

    # 모델 클래스 로드 및 Tokenizer 초기화
    # 모델에 맞는 토크나이저(Chat Template 포함)를 가져오기 위해 모델 클래스를 먼저 로드합니다.
    model_cls = MODEL_REGISTRY.get(cfg.model.type)

    if cfg.mode == "inference":
        logger.info("추론 모드 시작")
        log_gpu_status(logger)

        # 2-1. Model 초기화
        try:
            ollama_config = OmegaConf.to_container(cfg.get("ollama"), resolve=True)
        except Exception:
            raise ValueError("inference.ollama 설정을 불러오는 데 실패했습니다.")

        logger.debug(f"ollama_config: {ollama_config}")
        model = model_cls(
            model_name_or_path=cfg.model.get("model_name_or_path"),
            ollama=ollama_config,
        )

        # 2-2. 추론용 데이터셋 전처리 시작
        logger.info(f"추론용 데이터셋 전처리 시작")

        test_dataset_path = None
        try:
            test_dataset_path = cfg.inference.get("test_dataset_path")
            if not os.path.exists(test_dataset_path):
                raise ValueError("추론용 데이터셋 경로를 찾을 수 없습니다.")
        except Exception as e:
            raise ValueError(f"inference.test_dataset_path 설정을 불러오는 데 실패했습니다. {e}")

        test_dataset_cls = DATASET_REGISTRY.get(cfg.dataset.type)
        test_dataset = test_dataset_cls(test_dataset_path)
        logger.info(f"cfg inference {cfg.inference}")
        processed_test_dataset = test_dataset.preprocess(
            template=cfg.prompt.name,
            **cfg.dataset.preprocess.inference,
        )
        logger.info("추론용 데이터셋 전처리 완료")

        # 2-4. 추론 수행
        logger.info("모델 추론 시작")
        predictions = model.predict(dataset=processed_test_dataset["train"], **cfg.inference)

        # 2-5. 결과 출력 (일부만)
        logger.info(f"총 {len(predictions)}개 예측 완료")
        logger.info(f"예측 결과 샘플: {list(predictions.items())[:3]}")

        # 2-6. output.csv 저장
        try:
            output_path = Path(cfg.inference.get("output_path", "outputs/fallback"))
            os.makedirs(output_path, exist_ok=True)

            from datetime import datetime

            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            base_file_name = f"{current_config_name}_{timestamp}_output"
            file_name = f"{base_file_name}.csv"
            final_output_path = output_path / file_name

            # 중복 파일명 처리
            # # predictions 딕셔너리를 DataFrame으로 변환
            df_output = pd.DataFrame([{"id": id, "answer": answer} for id, answer in predictions.items()])
            df_output = df_output.sort_values("id")  # id 순서대로 정렬
            logger.info(f"결과를 {final_output_path}에 저장합니다.")
            df_output.to_csv(final_output_path, index=False)
            logger.info(f"결과 저장 완료: {final_output_path}")

            # 실행 시간 정보 저장
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60

            time_info_file = f"{base_file_name}.txt"
            time_info_path = output_path / time_info_file

            with open(time_info_path, "w", encoding="utf-8") as f:
                f.write(f"실행 시간: {minutes}분 {seconds:.2f}초\n")
                f.write(f"총 예측 개수: {len(predictions)}\n")
                f.write(f"설정 파일: {current_config_name}\n")

            logger.info(f"실행 시간 정보 저장 완료: {time_info_path}")
            logger.info(f"총 실행 시간: {minutes}분 {seconds:.2f}초")

            # 추론에 사용한 입력 데이터셋 저장
            input_data_file_name = f"{base_file_name}_input_data.csv"
            input_data_path = output_path / input_data_file_name

            # processed_test_dataset에서 DataFrame 생성
            input_df = pd.DataFrame(processed_test_dataset["train"])
            input_df.to_csv(input_data_path, index=False)
            logger.info(f"입력 데이터셋 저장 완료: {input_data_path}")

            logger.info(f"모든 추론이 종료되었습니다!")

        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {e}")
            raise

    elif cfg.mode == "evaluate":
        logger.info("평가 모드 시작")

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
        logger.info(f"평가 완료. 메트릭: {metrics}")


if __name__ == "__main__":
    main()
