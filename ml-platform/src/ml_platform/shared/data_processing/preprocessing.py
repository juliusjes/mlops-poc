import numpy as np
import torch


def add_batch_dim(x: torch.Tensor):
    x = x.unsqueeze(0)
    return x


def numpy_to_tensor(x: np.ndarray):
    return torch.from_numpy(x).float()