from typing import List, Union
from diffusers.utils import BaseOutput
from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class Fields2DPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        fields torch or numpy array of shape (batch_size, channels, height, width):
    """

    fields: Union[torch.tensor, np.ndarray]

def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device='cpu'):
    # modified from diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma