import random
import numpy as np
from src.utils.logger import get_logger
# import torch


def set_seed(seed: int):
    """
    재현성을 위해 난수 시드를 고정합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # logger = get_logger()
    # logger.info(f"✅ 난수 시드가 {seed}로 고정되었습니다.")
