import sys
import os
import copy
import torch
import math as mt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.inverse_utils import create_scatter_mask, ensemble_sample, edm_sampler
from dataloader.dataset_class import inverse_normalize_transform

@torch.no_grad()
def metric_func_2D(y_pred, y_true, mask=None):
    # Adapted from: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py
    # (B, C, H, W)
    '''
    y_pred: torch.Tensor, shape (B, C, H, W)
    y_true: torch.Tensor, shape (B, C, H, W)
    mask: torch.Tensor, shape (B, H, W), optional

    Returns: err_RMSE, err_nRMSE, err_CSV
    '''
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

    err_mean = torch.sqrt(mse) # (B, C)
    err_RMSE = torch.mean(err_mean, axis=0) # -> (C)
    err_nRMSE = torch.mean(err_mean / nrm, dim=0) # -> (C)
    err_CSV = torch.sqrt(csv_error) # (C)

    # Channel mean
    err_RMSE = torch.mean(err_RMSE, axis=0)
    err_nRMSE = torch.mean(err_nRMSE, axis=0)
    err_CSV = torch.mean(err_CSV, axis=0)
    
    return err_RMSE, err_nRMSE, err_CSV

def get_metrics_2D(val_dataset, pipeline=None, vt=None, vt_model=None, batch_size=64, ensemble_size=None,
                 sampler_kwargs=None, mask_kwargs=None, device='cpu', mode='edm',
                 inverse_transform=None):
    if mask_kwargs is None:
        mask_kwargs = {}
    if sampler_kwargs is None:
        sampler_kwargs = {}

    if inverse_transform == 'normalize':
        inverse_transform = inverse_normalize_transform

    if mode == 'edm':
        model = pipeline.unet
        noise_scheduler = copy.deepcopy(pipeline.scheduler)
    
    print(f'Calculating metrics for {mode}, ensemble size: {ensemble_size}')

    count = 0
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for step, y_true in tqdm(enumerate(val_loader), total=len(val_loader)):
        # (B, C, H, W)
        y_true = y_true.to(device)
        num_sample = y_true.shape[0]
        generator = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in range(count, count+num_sample)]
        if mode == 'vt':
            interpolated_fields = torch.empty_like(y_true)
        if vt is not None:
            scatter_mask = torch.empty_like(y_true)
            for b in range(num_sample):
                x_idx, y_idx = vt.structure_obs(generator=generator[b])
                if mode == 'vt':
                    grid_points = vt._get_grid_points(x_idx, y_idx)
                    for c in range(y_true.shape[1]):
                        field = y_true[b, c].cpu().numpy()[grid_points[:,1], grid_points[:,0]].flatten()
                        interpolated_fields[b, c] = torch.tensor(vt.interpolate(grid_points, field).reshape(y_true.shape[-2:]),
                                                            dtype=y_true.dtype, device=device)
                tmp_mask = tmp_mask = create_scatter_mask(y_true, x_idx=x_idx, y_idx=y_idx, device=device, **mask_kwargs)
                scatter_mask[b] = tmp_mask[0]
        else:
            scatter_mask = create_scatter_mask(y_true, generator=generator, device=device, **mask_kwargs)

        if mode == 'edm':
            if ensemble_size is not None:
                y_pred = torch.empty_like(y_true)
                for b in range(num_sample):
                    y_pred[b] = ensemble_sample(pipeline, ensemble_size, scatter_mask[[b]], sampler_kwargs=sampler_kwargs,
                                         class_labels=None, known_latents=y_true[[b]], sampler_type=mode, device=device).mean(dim=0)
            else:
                y_pred = edm_sampler(model, noise_scheduler, batch_size=num_sample, generator=generator, device=device,
                                        class_labels=None, mask=scatter_mask, known_latents=y_true, **sampler_kwargs)
        elif mode == 'pipeline':
            if ensemble_size is not None:
                y_pred = torch.empty_like(y_true)
                for b in range(num_sample):
                    y_pred[b] = ensemble_sample(pipeline, ensemble_size, scatter_mask[[b]], sampler_kwargs=sampler_kwargs,
                                         class_labels=None, known_latents=y_true[[b]], sampler_type=mode, device=device).mean(dim=0)
            else:
                y_pred = pipeline(batch_size=num_sample, generator=generator, 
                                        mask=tmp_mask, known_latents=y_true, return_dict=False, **sampler_kwargs)[0]

        elif mode == 'vt':
            y_pred = vt_model(interpolated_fields, return_dict=False)[0]

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

        count += num_sample
    
    return err_RMSE / (step + 1), err_nRMSE / (step + 1), err_CSV / (step + 1)