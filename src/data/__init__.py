# 레지스트리에 데이터셋을 등록하기 위해 import
# 각 데이터셋은 필요한 의존성이 없으면 건너뜀
# TODO 전력 로거로 변경
import logging

logger = logging.getLogger(__name__)

# Ollama 전용 데이터셋 (torch 불필요)
try:
    from src.data import qwen3_ollama_data
except ImportError as e:
    logger.debug(f"qwen3_data 로드 실패: {e}")

# 기존 데이터셋 (transformers 등 필요할 수 있음)
try:
    from src.data import baseline_data
except ImportError as e:
    logger.debug(f"baseline_data 로드 실패: {e}")
