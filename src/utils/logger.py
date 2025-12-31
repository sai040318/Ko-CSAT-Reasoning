import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig


# ANSI 색상 코드
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    # 레벨별 색상
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta
    # 추가 색상
    TIME = "\033[90m"  # Gray
    NAME = "\033[34m"  # Blue


class ColoredFormatter(logging.Formatter):
    """레벨별 색상을 적용하는 포맷터"""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def __init__(self, fmt: str, datefmt: str, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            # 레벨 색상
            level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
            record.levelname = f"{level_color}{record.levelname:<5}{LogColors.RESET}"
            # 시간 색상
            record.asctime = f"{LogColors.TIME}{self.formatTime(record, self.datefmt)}{LogColors.RESET}"
            # 메시지 포맷
            return f"{record.asctime} {record.levelname} {record.getMessage()}"
        else:
            return super().format(record)


# 로깅 레벨 문자열 → 정수 변환
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_logging_initialized = False


def setup_logging(level: Union[str, int] = logging.INFO, use_color: bool = True) -> None:
    """
    앱 시작 시 한 번만 호출하여 루트 로거를 설정

    Args:
        level: 로깅 레벨 (문자열 "DEBUG" 또는 정수 logging.DEBUG)
        use_color: 색상 사용 여부

    사용 예시:
        # run.py 시작부분에서
        from src.utils import setup_logging
        setup_logging(level="DEBUG", use_color=True)

    tqdm과 함께 사용 시:
        from tqdm.contrib.logging import logging_redirect_tqdm
        with logging_redirect_tqdm():
            for item in tqdm(items):
                logger.info("처리중...")
    """
    global _logging_initialized

    if _logging_initialized:
        return

    # 문자열 레벨을 정수로 변환
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), logging.INFO)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 기존 핸들러 제거 (Hydra 등에서 추가된 것)
    root_logger.handlers.clear()

    # 새 핸들러 추가
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        ColoredFormatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            use_color=use_color,
        )
    )
    root_logger.addHandler(handler)

    _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거를 반환 (핸들러 추가 없음, 루트로 전파)

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        logging.Logger: 로거 인스턴스

    사용 예시:
        from src.utils import get_logger
        logger = get_logger(__name__)
        logger.info("메시지")
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    실험 로깅을 위한 유틸리티 클래스

    실험 결과를 파일 및 콘솔에 로깅하고, config 및 메트릭을 저장

    tqdm 사용 시:
        with logging_redirect_tqdm():
            for item in tqdm(items):
                logger.info("처리중...")
    """

    def __init__(
        self, exp_name: Optional[str] = None, log_dir: str = "logs/experiments"
    ) -> None:
        # 실험 디렉토리 생성 (타임스탬프 기반)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name or f"exp_{timestamp}"
        self.exp_dir = Path(log_dir) / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 체크포인트 디렉토리 생성
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # 로깅 설정
        self.setup_logging()
        self.metrics = []

    def setup_logging(self) -> None:
        """로그 파일 및 콘솔 출력 설정"""
        log_file = self.exp_dir / "train.log"

        # 기존 핸들러 제거
        logger = logging.getLogger()
        logger.handlers = []

        # 새로운 핸들러 설정
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"실험 디렉토리: {self.exp_dir}")

    def save_config(self, cfg: DictConfig) -> None:
        """Hydra config를 YAML 파일로 저장"""
        config_path = self.exp_dir / "config.yaml"
        OmegaConf.save(cfg, config_path)
        self.logger.info(f"Config 저장 완료: {config_path}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """메트릭 로깅 (메모리에 저장)"""
        log_entry = {"step": step, **metrics}
        self.metrics.append(log_entry)

        if step is not None:
            metric_str = " | ".join(
                [
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in metrics.items()
                ]
            )
            self.logger.info(f"Step {step} - {metric_str}")

    def save_metrics(self) -> None:
        """메트릭을 JSON 파일로 저장"""
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(f"메트릭 저장 완료: {metrics_path}")

    def save_predictions(
        self, predictions: Dict[str, Any], filename: str = "predictions.json"
    ) -> None:
        """예측 결과를 JSON 파일로 저장"""
        pred_path = self.exp_dir / filename
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        self.logger.info(f"예측 결과 저장 완료: {pred_path}")

    def get_checkpoint_path(self, name: str = "best_model.pt") -> Path:
        """체크포인트 저장 경로 반환"""
        return self.checkpoint_dir / name

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)
