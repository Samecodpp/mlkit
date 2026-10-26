from functools import lru_cache

import torch

@lru_cache
def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
