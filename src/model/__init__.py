# 레지스트리에 모델을 등록하기 위해 import
# 각 모델은 필요한 의존성이 없으면 건너뜀 (torch, transformers 등)
from src.utils import get_logger

logger = get_logger(__name__)

__all__ = []

try:
    import unsloth

except ImportError as e:
    logger.debug(f"unsloth 라이브러리 로드 실패: {e}")


# Ollama 기반 모델 (torch 불필요)
try:
    from .qwen3_2507_thinking_model import Qwen3_2507ThinkingModel

    __all__.append("Qwen3_2507ThinkingModel")
except ImportError as e:
    logger.debug(f"Qwen3_2507_ThinkingModel 로드 실패 (의존성 누락): {e}")

# HuggingFace/PyTorch 기반 모델 (torch 필요)
try:
    from .baseline_model import BaselineModel

    __all__.append("BaselineModel")
except ImportError as e:
    logger.debug(f"BaselineModel 로드 실패 (torch 등 의존성 누락): {e}")

try:
    from .unsloth_model import UnslothModel

    __all__.append("UnslothModel")
except ImportError as e:
    logger.debug(f"UnslothModel 로드 실패 (의존성 누락): {e}")

try:
    from .qwen3_model import Qwen3Model

    __all__.append("Qwen3Model")
except ImportError as e:
    logger.debug(f"Qwen3Model 로드 실패 (의존성 누락): {e}")

try:
    from .exaone_model import ExaoneModel

    __all__.append("ExaoneModel")
except ImportError as e:
    logger.debug(f"ExaoneModel 로드 실패 (의존성 누락): {e}")
