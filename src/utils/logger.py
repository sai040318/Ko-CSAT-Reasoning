# src/utils/logger.py
"""
로깅 유틸 (기존 코드 특성 유지 + tqdm/모듈 로깅 이슈 완화)

핵심 설계(기존 의도 존중):
- ColoredFormatter(ANSI 컬러) 유지
- ExperimentLogger(실험 폴더/파일 저장) 유지
- "전역 로깅을 갈아엎지 않는다"가 기본값이지만,
  - tqdm 환경에서 로그가 진행바에 묻히는 문제를 줄이기 위해,
    기존 StreamHandler(stderr/stdout)만 선택적으로 TqdmLoggingHandler로 '교체'할 수 있게 함.
- Hydra/서드파티가 먼저 핸들러를 깔아도 "로그가 안 보이는" 상황을 막기 위해
  setup_logging이 최소한의 보정을 수행:
  - root level 설정
  - 필요 시 콘솔 핸들러 추가/교체
  - 중복 핸들러 추가 방지

권장 사용:
- 엔트리포인트(main) 진입 후 가능한 한 빨리 setup_logging() 호출
- 그 이후 get_logger(__name__)로 로거 생성/사용
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union

from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig


# ANSI 색상 코드
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta
    TIME = "\033[90m"  # Gray


class ColoredFormatter(logging.Formatter):
    """레벨별 색상을 적용하는 포맷터 (기존 유지)"""

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
        orig_levelname = record.levelname
        orig_asctime = getattr(record, "asctime", None)

        try:
            if self.use_color:
                level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
                record.levelname = f"{level_color}{record.levelname:<8}{LogColors.RESET}"
                record.asctime = f"{LogColors.TIME}{self.formatTime(record, self.datefmt)}{LogColors.RESET}"

            return super().format(record)

        finally:
            record.levelname = orig_levelname
            if orig_asctime is None:
                if hasattr(record, "asctime"):
                    delattr(record, "asctime")
            else:
                record.asctime = orig_asctime


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_logging_initialized = False
_last_logging_signature: Optional[tuple] = None


class TqdmLoggingHandler(logging.Handler):
    """
    tqdm 진행바와 충돌하지 않도록 tqdm.write로 출력하는 로깅 핸들러
    (기존 코드 유지 + 중복/교체 로직에서 쓰기 좋게 약간 보강)
    """

    def __init__(self, stream=None):
        super().__init__()
        self.stream = stream or sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # tqdm는 모듈/함수명이 같아도 여기서는 tqdm.write를 쓰는 게 목적
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        try:
            self.stream.flush()
        except Exception:
            pass


def _to_level(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    return LOG_LEVELS.get(str(level).upper(), logging.INFO)


def _has_equivalent_console_handler(root: logging.Logger, handler_type: type, stream) -> bool:
    for h in root.handlers:
        if isinstance(h, handler_type):
            if getattr(h, "stream", None) is stream:
                return True
    return False


def _replace_stream_handlers_with_tqdm(root: logging.Logger, formatter: logging.Formatter) -> None:
    """
    root에 있는 StreamHandler(stderr/stdout)를 tqdm 친화 핸들러로 교체.
    - RichHandler 등 다른 핸들러는 건드리지 않음
    - 교체 시 기존 핸들러 레벨을 그대로 계승
    """
    new_handlers = []
    replaced_any = False

    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stderr, sys.stdout):
            tq = TqdmLoggingHandler(stream=h.stream)
            tq.setLevel(h.level)
            tq.setFormatter(formatter)
            root.removeHandler(h)
            new_handlers.append(tq)
            replaced_any = True

    for nh in new_handlers:
        root.addHandler(nh)

    # 교체가 하나도 안 일어났다면 아무 것도 하지 않음
    _ = replaced_any


def setup_logging(
    level: Union[str, int] = "INFO",
    use_color: bool = True,
    *,
    prefer_tqdm: bool = True,
    replace_existing_stream_handlers: bool = True,
    force: bool = False,
    capture_warnings: bool = True,
) -> None:
    """
    앱 시작 시 한 번만 호출하여 로거 구성을 보정.

    기본 철학(기존 코드 존중):
    - root에 핸들러가 이미 있으면 "되도록 건드리지 않는다"
    - 다만 tqdm 환경에서 로그가 '안 보이는 것처럼' 묻히는 케이스가 흔하므로,
      기존 StreamHandler(stderr/stdout)만 선택적으로 TqdmLoggingHandler로 교체할 수 있다.

    파라미터:
    - prefer_tqdm: 콘솔 출력은 tqdm.write 기반을 선호
    - replace_existing_stream_handlers: root에 StreamHandler(stderr/stdout)가 있으면 교체
    - force: True면 root.handlers를 비우고(전역 재구성) 우리가 지정한 콘솔 핸들러 1개로 통일
    - capture_warnings: warnings를 logging으로 흡수
    """
    global _logging_initialized, _last_logging_signature

    lvl = _to_level(level)
    signature = (lvl, bool(use_color), bool(prefer_tqdm), bool(replace_existing_stream_handlers), bool(force))

    # 동일 설정으로 이미 초기화되었으면 재진입 방지
    if _logging_initialized and _last_logging_signature == signature:
        return

    root = logging.getLogger()
    root.setLevel(lvl)

    if capture_warnings:
        logging.captureWarnings(True)

    # 포맷(기존 스타일 유지: 시간/레벨/로거명)
    fmt = "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s - %(message)s"
    datefmt = "%H:%M:%S"
    formatter: logging.Formatter = ColoredFormatter(fmt=fmt, datefmt=datefmt, use_color=use_color)

    # force=True면 "전역 재구성"
    if force:
        # 완전 초기화: root의 기존 handler 제거
        for h in list(root.handlers):
            root.removeHandler(h)

        if prefer_tqdm:
            h = TqdmLoggingHandler(stream=sys.stderr)
        else:
            h = logging.StreamHandler(sys.stderr)

        h.setLevel(lvl)
        h.setFormatter(formatter)
        root.addHandler(h)

        _logging_initialized = True
        _last_logging_signature = signature
        return

    # force=False: 기존 구성 존중
    if not root.handlers:
        # root에 핸들러가 없으면 1개만 추가
        if prefer_tqdm:
            h = TqdmLoggingHandler(stream=sys.stderr)
        else:
            h = logging.StreamHandler(sys.stderr)

        h.setLevel(lvl)
        h.setFormatter(formatter)
        root.addHandler(h)

        _logging_initialized = True
        _last_logging_signature = signature
        return

    # 이미 핸들러가 있는 경우: 기본은 존중하되, tqdm 안정성을 위해 StreamHandler만 선택 교체 가능
    if prefer_tqdm and replace_existing_stream_handlers:
        _replace_stream_handlers_with_tqdm(root, formatter)

    # root 핸들러들의 포맷이 제각각인 경우도 있으니,
    # 최소한 "우리 핸들러"가 있다면 formatter가 적용되도록 보정
    for h in root.handlers:
        if isinstance(h, (TqdmLoggingHandler, logging.StreamHandler)) and getattr(h, "stream", None) in (
            sys.stderr,
            sys.stdout,
        ):
            if h.formatter is None or isinstance(h.formatter, logging.Formatter):
                # 기존 포맷을 유지하고 싶으면 여기서 setFormatter를 안 하면 되지만,
                # 모듈별 로그 가독성을 위해 기본 포맷을 통일하는 편이 실무에서 더 낫다.
                h.setFormatter(formatter)

    _logging_initialized = True
    _last_logging_signature = signature


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    모듈별 로거 반환 (핸들러 추가 없음, 루트로 전파)
    """
    return logging.getLogger(name)


def _resolve_log_base_dir() -> Path:
    """
    Hydra는 실행 시 작업 디렉토리를 run dir로 바꿀 수 있어 상대경로 로그가 꼬일 수 있다.
    가능하면 original_cwd를 기준으로 경로를 잡는다.
    """
    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd())
    except Exception:
        return Path.cwd()


def tune_third_party_log_levels(overrides: Optional[Dict[str, Union[int, str]]] = None) -> None:
    """
    서드파티 로거 소음 제어(선택).
    """
    defaults: Dict[str, Union[int, str]] = {
        "httpx": "WARNING",
        "httpcore": "WARNING",
        "urllib3": "WARNING",
        "datasets": "WARNING",
        "hydra": "INFO",
        "asyncio": "WARNING",
        "transformers": "WARNING",
    }
    if overrides:
        defaults.update(overrides)

    for name, lvl in defaults.items():
        logging.getLogger(name).setLevel(_to_level(lvl))


class ExperimentLogger:
    """
    실험 로깅 유틸리티 (기존 구조 유지)

    - 전역(root) 로깅 구성은 건드리지 않음
    - exp 전용 로거에 FileHandler만 추가하여 파일 저장
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
        self._setup_file_logging()

    def _setup_file_logging(self) -> None:
        log_file = self.exp_dir / "train.log"

        logger_name = f"{__name__}.experiment.{self.exp_name}"
        self.logger = logging.getLogger(logger_name)
        # 파일 로그는 기본 INFO (기존 유지). 필요하면 외부에서 setLevel로 올려도 됨.
        self.logger.setLevel(logging.INFO)

        # 중복 FileHandler 방지
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

        # 콘솔 출력은 루트가 담당 (propagate=True 유지)
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
