import sys
import os
import copy
import torch
import math as mt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.inverse_utils import create_scatter_mask, ensemble_sample, colored_noise
from dataloader.dataset_class import inverse_normalize_transform

@torch.no_grad()
def metric_func_2D(y_pred, y_true, mask=None, channel_mean=True):
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
        csv_error = torch.mean((torch.sum(y_pred * unknown_mask, dim=[-2, -1]) - torch.sum(y_true * unknown_mask, dim=[-2, -1]))**2, dim=0) 
    else:
        mse = torch.mean((y_pred - y_true)**2, dim=[-2, -1]) 
        nrm = torch.sqrt(torch.mean(y_true**2, dim=[-2, -1])) 
        csv_error = torch.mean((torch.sum(y_pred, dim=[-2, -1]) - torch.sum(y_true, dim=[-2, -1]))**2, dim=0)

    err_mean = torch.sqrt(mse) # (B, C)
    err_RMSE = torch.mean(err_mean, axis=0) # -> (C)
    err_nRMSE = torch.mean(err_mean / nrm, dim=0) # -> (C)
    err_CSV = torch.sqrt(csv_error) # (C)
    if mask is not None:
        err_CSV /= torch.mean(torch.sum(unknown_mask, dim=[-2, -1]), dim=0)
    else:
        err_CSV /= (y_true.shape[-2] * y_true.shape[-1])

    # Channel mean
    if channel_mean:
        err_RMSE = torch.mean(err_RMSE, axis=0)
        err_nRMSE = torch.mean(err_nRMSE, axis=0)
        err_CSV = torch.mean(err_CSV, axis=0)
    
    return err_RMSE, err_nRMSE, err_CSV

@torch.no_grad()
def get_metrics_2D(val_dataset, pipeline=None, vt=None, vt_model=None, batch_size=64, ensemble_size=25,
                 sampler_kwargs=None, mask_kwargs=None, known_channels=None, device='cpu', mode='edm', #'edm', 'pipeline', 'vt', 'mean'
                 conditioning_type = None, #xattn, cfg
                 inverse_transform=None, inverse_transform_args=None, channel_mean=True,
                 structure_sampling=False, noise_level=0, noise_type='white', # 'white', 'pink', 'red', 'blue', 'purple'
                 verbose=False):

    if inverse_transform == 'normalize':
        inverse_transform = inverse_normalize_transform

    if mode != 'mean':
        if mask_kwargs is None:
            mask_kwargs = {}
        if sampler_kwargs is None:
            sampler_kwargs = {}

        if mode == 'edm':
            model = pipeline.unet
            noise_scheduler = copy.deepcopy(pipeline.scheduler)

        if "x_idx" in mask_kwargs and "y_idx" in mask_kwargs:
            print("Using structure sampling, x_idx and y_idx are provided.")
            assert structure_sampling, "Structure sampling must be enabled when x_idx and y_idx are provided."
            x_idx, y_idx = mask_kwargs["x_idx"], mask_kwargs["y_idx"]
            mask_kwargs.pop("x_idx")
            mask_kwargs.pop("y_idx")
            ignore_idx_check = True
        else:
            if structure_sampling:
                print("Using structure sampling, x_idx and y_idx are not provided, selecting random points from each grid.")
            else:
                print("Not using structure sampling. points are uniformly sampled.")
            ignore_idx_check = False

        
        print(f'Calculating metrics for {mode}, ensemble size: {ensemble_size}')
    else:
        print('Calculating metrics for mean, all other arguments are ignored.')
    
    if structure_sampling and noise_type != 'white':
        Warning("Colored noise is not supported with structure sampling. Ignoring noise_type.")

    count = 0
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for step, y_true in tqdm(enumerate(val_loader), total=len(val_loader)):
        # (B, C, H, W)
        y_true = y_true.to(device)
        num_sample, C, _, _ = y_true.shape
        if mode != 'mean':
            generator = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in range(count, count+num_sample)]
            if mode == 'vt':
                in_channels = vt_model.config.in_channels
            else:
                in_channels = C if known_channels is None else len(known_channels)
            interpolated_fields = torch.empty((num_sample, in_channels, y_true.shape[-2], y_true.shape[-1]), device=device, dtype=y_true.dtype)
            if structure_sampling:
                scatter_mask = torch.empty_like(y_true)

                for b in range(num_sample):
                    if ("x_idx" not in mask_kwargs and "y_idx" not in mask_kwargs) and not ignore_idx_check:
                        x_idx, y_idx = vt.structure_obs(generator=generator[b])

                    grid_points = vt._get_grid_points(x_idx, y_idx, generator=generator[b])

                    for idx, known_channel in enumerate(range(C) if known_channels is None else known_channels):
                        field = y_true[b, known_channel][grid_points[:,1], grid_points[:,0]].flatten()
                        if noise_level > 0:
                            # drop support for colored noise in this case
                            noise = randn_tensor(field.shape, device=device)
                            field += noise * noise_level * field # y_obs = y_true + noise_level * y_true
                        interpolated_values = vt.interpolate(grid_points, field)
                        interpolated_fields[b, idx] = torch.tensor(interpolated_values, 
                                                                dtype=y_true.dtype, 
                                                                device=device)
                    tmp_mask = tmp_mask = create_scatter_mask(y_true, x_idx=x_idx, y_idx=y_idx, device=device, **mask_kwargs)
                    scatter_mask[b] = tmp_mask[0]
            else:
                scatter_mask = create_scatter_mask(y_true, generator=generator, device=device, **mask_kwargs)
                tmp_y_true = y_true.clone()
                if noise_level > 0:
                    if noise_type == 'white':
                        noise = randn_tensor(tmp_y_true.shape, device=device)
                    else:
                        noise = colored_noise(tmp_y_true.shape, noise_type=noise_type, device=device)
                    tmp_y_true += noise * noise_level * tmp_y_true # y_obs = y_true + noise_level * y_true
                interpolated_fields = vt(known_fields=tmp_y_true, mask=scatter_mask)

            if mode == 'edm':
                y_pred = torch.empty_like(y_true)
                for b in range(num_sample):
                    y_pred[b] = ensemble_sample(pipeline, ensemble_size, scatter_mask[[b]], sampler_kwargs=sampler_kwargs, conditioning_type=conditioning_type,
                                            class_labels=None, known_latents=interpolated_fields[[b]], sampler_type=mode, device=device).mean(dim=0)
            elif mode == 'pipeline':
                y_pred = torch.empty_like(y_true)
                for b in range(num_sample):
                    y_pred[b] = ensemble_sample(pipeline, ensemble_size, scatter_mask[[b]], sampler_kwargs=sampler_kwargs, conditioning_type=conditioning_type,
                                            class_labels=None, known_latents=interpolated_fields[[b]], sampler_type=mode, device=device).mean(dim=0)

            elif mode == 'vt':
                y_pred = vt_model(interpolated_fields, return_dict=False)[0]

            else:
                raise NotImplementedError(f'Mode: {mode} not implemented.')

            if inverse_transform is not None:
                y_pred = inverse_transform(y_pred, **inverse_transform_args)
                y_true = inverse_transform(y_true, **inverse_transform_args)
        else:
            y_true = inverse_transform(y_true, **inverse_transform_args)
            y_pred = torch.ones_like(y_true) * repeat(torch.tensor(inverse_transform_args['mean'], device=device), 'c -> b c 1 1', b=num_sample)
            

        _err_RMSE, _err_nRMSE, _err_CSV = metric_func_2D(y_pred, y_true, 
                                                         mask=scatter_mask if mode != 'mean' else None,
                                                         channel_mean=channel_mean)

        if step == 0:
            err_RMSE = _err_RMSE * num_sample
            err_nRMSE = _err_nRMSE * num_sample
            err_CSV = _err_CSV * num_sample
        else:
            err_RMSE += _err_RMSE * num_sample
            err_nRMSE += _err_nRMSE * num_sample
            err_CSV += _err_CSV * num_sample

        count += num_sample

        if verbose:
            print(f'RMSE: {err_RMSE / count}, nRMSE: {err_nRMSE / count}, CSV: {err_CSV / count}')
    
    return err_RMSE / count, err_nRMSE / count, err_CSV / count