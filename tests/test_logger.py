"""
logger.py sanity check 테스트

테스트 항목:
1. setup_logging 함수 (루트 로거 설정)
2. get_logger 함수 (모듈별 로거 반환)
3. ColoredFormatter 색상 적용
4. tqdm과의 호환성 (logging_redirect_tqdm)
5. ExperimentLogger 클래스 기본 동작
"""

import logging
import tempfile
import shutil
from pathlib import Path

import pytest
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.utils.logger import (
    get_logger,
    setup_logging,
    ExperimentLogger,
    ColoredFormatter,
    LogColors,
    LOG_LEVELS,
    _logging_initialized,
)
import src.utils.logger as logger_module


@pytest.fixture(autouse=True)
def reset_logging():
    """각 테스트 전후로 로깅 상태 초기화"""
    # 테스트 전 상태 저장
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    # _logging_initialized 리셋
    logger_module._logging_initialized = False

    yield

    # 테스트 후 상태 복원
    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)
    logger_module._logging_initialized = False


class TestSetupLogging:
    """setup_logging 함수 테스트"""

    def test_setup_logging_sets_root_handler(self):
        """setup_logging이 루트 로거에 핸들러를 추가하는지 확인"""
        setup_logging(level="INFO")
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_sets_level_from_string(self):
        """문자열 레벨이 정수로 변환되는지 확인"""
        setup_logging(level="DEBUG")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_sets_level_from_int(self):
        """정수 레벨이 그대로 적용되는지 확인"""
        setup_logging(level=logging.WARNING)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_setup_logging_only_once(self):
        """setup_logging이 한 번만 실행되는지 확인"""
        setup_logging(level="DEBUG")
        root_logger = logging.getLogger()
        handler_count = len(root_logger.handlers)

        # 두 번째 호출
        setup_logging(level="INFO")  # 다른 레벨로 시도

        # 핸들러가 추가되지 않아야 함
        assert len(root_logger.handlers) == handler_count

    def test_setup_logging_uses_colored_formatter(self):
        """ColoredFormatter가 사용되는지 확인"""
        setup_logging(level="INFO", use_color=True)
        root_logger = logging.getLogger()
        formatter = root_logger.handlers[0].formatter
        assert isinstance(formatter, ColoredFormatter)


class TestGetLogger:
    """get_logger 함수 테스트"""

    def test_get_logger_returns_logger(self):
        """get_logger가 Logger 인스턴스를 반환하는지 확인"""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_returns_same_instance(self):
        """같은 이름으로 호출 시 동일한 로거 반환 확인"""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """다른 이름으로 호출 시 다른 로거 반환 확인"""
        logger1 = get_logger("name_a")
        logger2 = get_logger("name_b")
        assert logger1 is not logger2

    def test_get_logger_no_handlers(self):
        """get_logger가 핸들러를 추가하지 않는지 확인"""
        logger = get_logger("no_handler_test")
        # 개별 로거에는 핸들러가 없어야 함 (루트로 전파)
        assert len(logger.handlers) == 0

    def test_get_logger_propagates_to_root(self):
        """로거가 루트로 전파되는지 확인 (propagate=True)"""
        logger = get_logger("propagate_test")
        assert logger.propagate is True


class TestLogLevels:
    """LOG_LEVELS 상수 테스트"""

    def test_log_levels_defined(self):
        """모든 로깅 레벨이 정의되어 있는지 확인"""
        assert LOG_LEVELS["DEBUG"] == logging.DEBUG
        assert LOG_LEVELS["INFO"] == logging.INFO
        assert LOG_LEVELS["WARNING"] == logging.WARNING
        assert LOG_LEVELS["ERROR"] == logging.ERROR
        assert LOG_LEVELS["CRITICAL"] == logging.CRITICAL


class TestColoredFormatter:
    """ColoredFormatter 클래스 테스트"""

    def test_colored_formatter_level_colors(self):
        """레벨별 색상이 정의되어 있는지 확인"""
        assert logging.DEBUG in ColoredFormatter.LEVEL_COLORS
        assert logging.INFO in ColoredFormatter.LEVEL_COLORS
        assert logging.WARNING in ColoredFormatter.LEVEL_COLORS
        assert logging.ERROR in ColoredFormatter.LEVEL_COLORS
        assert logging.CRITICAL in ColoredFormatter.LEVEL_COLORS

    def test_log_colors_defined(self):
        """ANSI 색상 코드가 정의되어 있는지 확인"""
        assert LogColors.RESET == "\033[0m"
        assert LogColors.DEBUG == "\033[36m"  # Cyan
        assert LogColors.INFO == "\033[32m"  # Green
        assert LogColors.WARNING == "\033[33m"  # Yellow
        assert LogColors.ERROR == "\033[31m"  # Red

    def test_colored_formatter_no_color_mode(self):
        """use_color=False일 때 색상 없이 포맷되는지 확인"""
        formatter = ColoredFormatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            use_color=False,
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        # ANSI 코드가 포함되지 않아야 함
        assert "\033[" not in formatted
        assert "Test message" in formatted


class TestLoggerWithTqdm:
    """tqdm과의 호환성 테스트"""

    def test_tqdm_with_logging_redirect(self, capfd):
        """logging_redirect_tqdm과 함께 사용 시 로그가 올바르게 출력되는지 확인"""
        setup_logging(level="INFO")
        logger = get_logger("test_tqdm_compat")

        items = range(5)
        log_messages = []

        with logging_redirect_tqdm():
            for i in tqdm(items, desc="Test Progress"):
                message = f"Processing item {i}"
                logger.info(message)
                log_messages.append(message)

        # stderr 캡처 (tqdm과 logging 모두 stderr 사용)
        captured = capfd.readouterr()
        stderr_output = captured.err

        # 로그 메시지가 출력에 포함되어 있는지 확인
        for msg in log_messages:
            assert msg in stderr_output, f"Log message '{msg}' not found in output"

    def test_tqdm_without_redirect_comparison(self, capfd):
        """logging_redirect_tqdm 사용 유무에 따른 차이 확인"""
        setup_logging(level="INFO")
        logger = get_logger("test_tqdm_no_redirect")

        # logging_redirect_tqdm 사용
        with logging_redirect_tqdm():
            for i in tqdm(range(3), desc="With redirect"):
                logger.info(f"With redirect: {i}")

        captured_with = capfd.readouterr()

        # 로그가 출력되었는지 확인
        assert "With redirect: 0" in captured_with.err
        assert "With redirect: 1" in captured_with.err
        assert "With redirect: 2" in captured_with.err

    def test_nested_tqdm_with_logging(self, capfd):
        """중첩된 tqdm 루프에서도 로깅이 올바르게 동작하는지 확인"""
        setup_logging(level="INFO")
        logger = get_logger("test_nested_tqdm")

        with logging_redirect_tqdm():
            for i in tqdm(range(2), desc="Outer"):
                logger.info(f"Outer loop: {i}")
                for j in tqdm(range(2), desc="Inner", leave=False):
                    logger.debug(f"Inner loop: {i}-{j}")  # DEBUG는 출력 안됨 (레벨이 INFO)

        captured = capfd.readouterr()
        assert "Outer loop: 0" in captured.err
        assert "Outer loop: 1" in captured.err


class TestExperimentLogger:
    """ExperimentLogger 클래스 테스트"""

    @pytest.fixture
    def temp_log_dir(self):
        """임시 로그 디렉토리 생성 및 정리"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_experiment_logger_creates_directory(self, temp_log_dir):
        """ExperimentLogger가 디렉토리를 생성하는지 확인"""
        exp_logger = ExperimentLogger(exp_name="test_exp", log_dir=temp_log_dir)
        assert exp_logger.exp_dir.exists()
        assert exp_logger.checkpoint_dir.exists()

    def test_experiment_logger_creates_log_file(self, temp_log_dir):
        """로그 파일이 생성되는지 확인"""
        exp_logger = ExperimentLogger(exp_name="test_log_file", log_dir=temp_log_dir)
        log_file = exp_logger.exp_dir / "train.log"
        assert log_file.exists()

    def test_experiment_logger_log_methods(self, temp_log_dir):
        """로깅 메서드가 정상 동작하는지 확인"""
        exp_logger = ExperimentLogger(exp_name="test_methods", log_dir=temp_log_dir)

        # 각 레벨의 로깅이 에러 없이 동작하는지 확인
        exp_logger.info("Info message")
        exp_logger.debug("Debug message")
        exp_logger.warning("Warning message")
        exp_logger.error("Error message")

        # 로그 파일에 기록되었는지 확인
        log_file = exp_logger.exp_dir / "train.log"
        log_content = log_file.read_text(encoding="utf-8")
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content

    def test_experiment_logger_log_metrics(self, temp_log_dir):
        """메트릭 로깅이 정상 동작하는지 확인"""
        exp_logger = ExperimentLogger(exp_name="test_metrics", log_dir=temp_log_dir)

        exp_logger.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
        exp_logger.log_metrics({"loss": 0.3, "accuracy": 0.9}, step=2)

        assert len(exp_logger.metrics) == 2
        assert exp_logger.metrics[0]["loss"] == 0.5
        assert exp_logger.metrics[1]["step"] == 2

    def test_experiment_logger_save_metrics(self, temp_log_dir):
        """메트릭 저장이 정상 동작하는지 확인"""
        exp_logger = ExperimentLogger(exp_name="test_save_metrics", log_dir=temp_log_dir)

        exp_logger.log_metrics({"loss": 0.5}, step=1)
        exp_logger.save_metrics()

        metrics_file = exp_logger.exp_dir / "metrics.json"
        assert metrics_file.exists()

        import json

        with open(metrics_file, "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
        assert len(saved_metrics) == 1
        assert saved_metrics[0]["loss"] == 0.5

    def test_experiment_logger_save_predictions(self, temp_log_dir):
        """예측 결과 저장이 정상 동작하는지 확인"""
        exp_logger = ExperimentLogger(exp_name="test_predictions", log_dir=temp_log_dir)

        predictions = {"id1": 1, "id2": 3, "id3": 5}
        exp_logger.save_predictions(predictions)

        pred_file = exp_logger.exp_dir / "predictions.json"
        assert pred_file.exists()

    def test_experiment_logger_get_checkpoint_path(self, temp_log_dir):
        """체크포인트 경로 반환이 정상 동작하는지 확인"""
        exp_logger = ExperimentLogger(exp_name="test_ckpt", log_dir=temp_log_dir)

        ckpt_path = exp_logger.get_checkpoint_path("model.pt")
        assert ckpt_path == exp_logger.checkpoint_dir / "model.pt"


class TestLoggerIntegration:
    """통합 테스트: 실제 사용 시나리오"""

    def test_setup_then_get_logger(self, capfd):
        """setup_logging 후 get_logger로 로깅하는 전체 흐름 테스트"""
        setup_logging(level="DEBUG", use_color=False)
        logger = get_logger("integration_test")

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        captured = capfd.readouterr()
        assert "Debug message" in captured.err
        assert "Info message" in captured.err
        assert "Warning message" in captured.err

    def test_multiple_modules_same_root(self, capfd):
        """여러 모듈이 같은 루트 로거를 공유하는지 확인"""
        setup_logging(level="INFO", use_color=False)

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        logger1.info("From module1")
        logger2.info("From module2")

        captured = capfd.readouterr()
        assert "From module1" in captured.err
        assert "From module2" in captured.err

    def test_tqdm_logging_preserves_progress_bar(self, capfd):
        """tqdm 진행바가 로깅과 함께 올바르게 표시되는지 확인"""
        setup_logging(level="INFO")
        logger = get_logger("test_progress_bar")

        with logging_redirect_tqdm():
            pbar = tqdm(range(10), desc="Test")
            for i in pbar:
                if i == 5:
                    logger.info("Halfway done!")
            pbar.close()

        captured = capfd.readouterr()
        assert "Halfway done!" in captured.err
        # tqdm 진행바 관련 출력도 있어야 함
        assert "Test" in captured.err or "100%" in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
