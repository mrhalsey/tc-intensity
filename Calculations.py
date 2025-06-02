import torch
import numpy as np
from torch import Tensor

tensor = Tensor

def normalize(x: tensor) -> Tensor:
    tensor = x
    valid_mask = ~torch.isnan(tensor)
    valid_tensor = tensor[valid_mask]

    min_val = valid_tensor.min()
    max_val = valid_tensor.max()

    normalized_tensor = (0.1 * (max_val - tensor) + 0.9 * (tensor - min_val)) / (max_val - min_val)
    return normalized_tensor