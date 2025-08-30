import numpy as np
import numpy.typing as npt
import random
import torch

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_indices = np.random.randint(
        0,
        len(dataset) - context_length - 1,
        (batch_size, 1),
    )
    indices = starting_indices + np.arange(context_length+1).reshape(1, -1)
    sample = torch.tensor(dataset[indices], device=device)
    return sample[:,:-1], sample[:,1:]