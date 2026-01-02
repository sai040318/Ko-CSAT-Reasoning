# src/utils/logger.py
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
        # 다른 핸들러/포맷터에도 같은 record가 전달될 수 있으므로,
        # record를 수정했다면 반드시 복원해서 부작용을 없앤다.
        orig_levelname = record.levelname
        orig_asctime = getattr(record, "asctime", None)

        try:
            if self.use_color:
                # 레벨 색상
                level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
                record.levelname = f"{level_color}{record.levelname:<5}{LogColors.RESET}"

                # 시간 색상
                record.asctime = f"{LogColors.TIME}{self.formatTime(record, self.datefmt)}{LogColors.RESET}"

            return super().format(record)

        finally:
            record.levelname = orig_levelname
            if orig_asctime is None:
                if hasattr(record, "asctime"):
                    delattr(record, "asctime")
            else:
                record.asctime = orig_asctime


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
    앱 시작 시 한 번만 호출하여 루트 로거를 설정.

    옵션 1(best practice) 핵심:
      - 이미 Hydra/pytest 등 다른 프레임워크가 루트 핸들러를 구성했다면 절대 건드리지 않는다.
      - 루트에 핸들러가 없을 때만, 컬러 콘솔 핸들러를 추가한다.
      - root_logger.handlers.clear() / logger.handlers=[] / logging.basicConfig(...) 같은
        "전역 로깅 갈아엎기" 동작은 하지 않는다.
    """
    global _logging_initialized
    if _logging_initialized:
        return

    # 문자열 레벨을 정수로 변환
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 이미 누군가(Hydra 등)가 로깅을 구성했다면 존중하고 종료
    if root_logger.handlers:
        _logging_initialized = True
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    # ✅ 모듈 추적을 위해 %(name)s 포함 (중요)
    handler.setFormatter(
        ColoredFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%H:%M:%S",
            use_color=use_color,
        )
    )
    root_logger.addHandler(handler)

    _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 반환 (핸들러 추가 없음, 루트로 전파)

    사용 예:
      logger = get_logger(__name__)
      logger.info("message")
    """
    return logging.getLogger(name)


def _resolve_log_base_dir() -> Path:
    """
    Hydra는 기본적으로 실행 시 작업 디렉토리를 run dir로 바꾸는 경우가 있어
    상대경로 로그가 의도치 않은 위치로 갈 수 있다.
    가능한 경우 original_cwd를 기준으로 로그 경로를 잡는다.
    """
    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd())
    except Exception:
        return Path.cwd()


def tune_third_party_log_levels(overrides: Optional[Dict[str, Union[int, str]]] = None) -> None:
    """
    서드파티 로거 소음 제어(선택).
    필요 없으면 호출하지 않아도 됨.
    """
    defaults: Dict[str, Union[int, str]] = {
        "httpx": "WARNING",
        "httpcore": "WARNING",
        "urllib3": "WARNING",
        "datasets": "WARNING",
        "hydra": "INFO",
        "asyncio": "WARNING",
    }
    if overrides:
        defaults.update(overrides)

    for name, lvl in defaults.items():
        if isinstance(lvl, str):
            lvl = LOG_LEVELS.get(lvl.upper(), logging.INFO)
        logging.getLogger(name).setLevel(lvl)


class ExperimentLogger:
    """
    실험 로깅 유틸리티

    변경 최소화:
      - 기존 API 유지 (save_config/log_metrics/save_metrics/save_predictions 등 그대로)
      - 다만 전역(root) 로깅을 "삭제/재구성"하지 않고
        실험 전용 로거에 FileHandler만 추가해서 파일 저장을 구현
    """

    def __init__(self, exp_name: Optional[str] = None, log_dir: str = "logs/experiments") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name or f"exp_{timestamp}"

        base_dir = _resolve_log_base_dir()
        self.exp_dir = (base_dir / log_dir / self.exp_name).resolve()
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.metrics = []
        self.setup_logging()

    def setup_logging(self) -> None:
        """실험 로그 파일 핸들러만 추가 (전역 로깅 구성은 절대 건드리지 않음)"""
        log_file = self.exp_dir / "train.log"

        # exp별 로거 분리 (충돌/중복 방지)
        logger_name = f"{__name__}.experiment.{self.exp_name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # 동일 파일에 대한 FileHandler 중복 추가 방지
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_file):
                break
        else:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(fh)

        # 콘솔 출력은 루트(Hydra 또는 setup_logging)가 담당하도록 propagate 유지
        # 만약 실험 로거 메시지를 콘솔에 중복 출력시키고 싶지 않다면 False로 바꾸면 됨.
        self.logger.propagate = True

        self.logger.info(f"실험 디렉토리: {self.exp_dir}")

    def save_config(self, cfg: DictConfig) -> None:
        config_path = self.exp_dir / "config.yaml"
        OmegaConf.save(cfg, config_path)
        self.logger.info(f"Config 저장 완료: {config_path}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        log_entry = {"step": step, **metrics}
        self.metrics.append(log_entry)

        if step is not None:
            metric_str = " | ".join(
                [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
            )
            self.logger.info(f"Step {step} - {metric_str}")

    def save_metrics(self) -> None:
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(f"메트릭 저장 완료: {metrics_path}")

    def save_predictions(self, predictions: Dict[str, Any], filename: str = "predictions.json") -> None:
        pred_path = self.exp_dir / filename
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        self.logger.info(f"예측 결과 저장 완료: {pred_path}")

    def get_checkpoint_path(self, name: str = "best_model.pt") -> Path:
        return self.checkpoint_dir / name

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)
