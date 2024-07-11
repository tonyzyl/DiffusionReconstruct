import sys
import os
import torch
import math as mt
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.inverse_utils import create_scatter_mask
from dataloader.dataset_class import inverse_normalize_transform

def metric_func_2D(y_pred, y_true, mask=None):
    # Adapted from: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py
    # (B, C, H, W)
    if mask is not None:
        unknown_mask = (1 - mask).float()
        num_unknowns = torch.sum(unknown_mask, dim=[-2, -1], keepdim=True)
        assert torch.all(num_unknowns) > 0, "All values are known. Cannot compute with mask."
        mse = torch.sum(((y_pred - y_true)**2) * unknown_mask, dim=[-2, -1], keepdim=True) / num_unknowns
        mse = rearrange(mse, 'b c 1 1 -> b c')
        nrm = torch.sqrt(torch.sum((y_true**2) * unknown_mask, dim=[-2, -1], keepdim=True) / num_unknowns)
        nrm = rearrange(nrm, 'b c 1 1 -> b c')
        csv_error = torch.mean((torch.sum(y_pred * unknown_mask, dim=[-2, -1]) - torch.sum(y_true * unknown_mask, dim=[-2, -1]))**2, dim=0) / torch.mean(unknown_mask)
    else:
        mse = torch.mean((y_pred - y_true)**2, dim=[-2, -1])
        nrm = torch.sqrt(torch.mean(y_true**2, dim=[-2, -1]))
        csv_error = torch.mean((torch.sum(y_pred, dim=[-2, -1]) - torch.sum(y_true, dim=[-2, -1]))**2, dim=0)

    err_mean = torch.sqrt(mse)
    err_RMSE = torch.mean(err_mean, axis=0)
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)
    err_CSV = torch.sqrt(csv_error)
    
    # Channel mean
    err_RMSE = torch.mean(err_RMSE, axis=0)
    err_nRMSE = torch.mean(err_nRMSE, axis=0)
    err_CSV = torch.mean(err_CSV, axis=0)
    
    return err_RMSE, err_nRMSE, err_CSV

def get_metrics_2D(val_dataset, model, noise_scheduler=None, sampler=None, batch_size=64,
                 sampler_config=None, mask_config=None, generator=None, device='cpu', mode='edm',
                 inverse_transform=None):
    # TODO: Add exclusion of mask (known values)
    if mask_config is None:
        mask_config = {}
    if sampler_config is None:
        sampler_config = {}

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if inverse_transform == 'normalize':
        inverse_transform = inverse_normalize_transform
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for step, y_true in tqdm(enumerate(val_loader)):
        # (B, C, H, W)
        latents = randn_tensor(y_true.shape, device=device, generator=generator, dtype=model.dtype)
        if mode == 'edm':
            scatter_mask = create_scatter_mask(y_true, generator=generator, device=device, **mask_config)
            y_pred = sampler(model, noise_scheduler, latents=latents, class_labels=None, known_latents=y_true*scatter_mask,
                                mask=scatter_mask, return_trajectory=False, **sampler_config)

        else:
            raise NotImplementedError(f'Mode: {mode} not implemented.')

        if inverse_transform is not None:
            y_pred = inverse_transform(y_pred, **val_dataset.transform_args)
            y_true = inverse_transform(y_true, **val_dataset.transform_args)

        _err_RMSE, _err_nRMSE, _err_CSV = metric_func_2D(y_pred, y_true, mask=scatter_mask)

        if step == 0:
            err_RMSE = _err_RMSE
            err_nRMSE = _err_nRMSE
            err_CSV = _err_CSV
        else:
            err_RMSE += _err_RMSE
            err_nRMSE += _err_nRMSE
            err_CSV += _err_CSV
    
    return err_RMSE / (step + 1), err_nRMSE / (step + 1), err_CSV / (step + 1)