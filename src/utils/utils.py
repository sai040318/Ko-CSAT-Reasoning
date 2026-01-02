import random
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_torch():
    """
    torch는 optional dependency로 취급.
    - 설치되어 있으면 torch 모듈을 반환
    - 없거나 로드 실패하면 None 반환
    """
    try:
        import torch

        return torch
    except Exception as e:
        logger.debug(f"torch 라이브러리 로드 실패(Seed 고정은 python/numpy만 수행): {e}")
        return None


def set_seed(seed: int):
    """
    재현성을 위해 난수 시드를 고정합니다.
    인터페이스(set_seed(seed))는 유지하면서,
    torch가 있을 때만 torch seed를 추가로 고정합니다.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch = _get_torch()
    if torch is None:
        logger.info(f"torch 라이브러리가 없어 python/numpy만 시드 고정합니다.")
        logger.info(f"난수 시드가 {seed}로 고정되었습니다. (python/numpy only)")
        return

    # torch seed 고정
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시

    # 필요하면 아래 옵션을 “항상 켠다” 대신,
    # 프로젝트 정책에 맞춰 별도 함수/설정에서 제어하는 것을 권장합니다.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    logger.info(f"✅ 난수 시드가 {seed}로 고정되었습니다. (python/numpy/torch)")
