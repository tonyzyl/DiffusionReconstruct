import matplotlib.pyplot as plt
import torch
from diffusers.utils import make_image_grid
from diffusers.utils.torch_utils import randn_tensor
import os
import numpy as np
from einops import repeat
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
import copy

#'''
# differences are negligible, but this one is faster
def create_scatter_mask(tensor, channels=None, ratio=0.1, x_idx=None, y_idx=None, generator=None, device=None):
    '''
    return a mask that has the same shape as the input tensor, if multiple channels are specified, the same mask will be applied to all channels
    tensor: torch.Tensor
    channels: list of ints, denote the idx of known channels, default None. If None, all channels are masked
    ratio: float, default 0.1. The ratio of known elements
    x_idx, y_idx: int, default None. If not None, the mask will be applied to the specified indices. OrientationL (0,0) is the top left corner

    return: torch.Tensor (B, C, H, W)
    '''
    if device is None:
        device = tensor.device
    B, C, H, W = tensor.shape
    if channels is None:
        channels = torch.arange(C, device=device)  # Ensure the same device as the input tensor
    else:
        channels = torch.tensor(channels, device=device)  # Ensure the same device as the input tensor

    # Create a random mask for all elements
    if x_idx is not None and y_idx is not None:
        mask = torch.zeros(B, 1, H, W, device=device)
        mask[:, :, y_idx, x_idx] = 1
        if len(channels) > 1:
            mask = repeat(mask, 'B 1 H W -> B C H W', C=len(channels))
    else:
        mask = torch.rand(B, 1, H * W, device=device, generator=generator) < ratio
        if len(channels) > 1:
            mask = repeat(mask, 'B 1 N -> B C N', C=len(channels))

    # Initialize the final mask with zeros
    final_mask = torch.zeros_like(tensor)
    # Convert mask to the same dtype as final_mask
    mask = mask.type_as(final_mask)
    # Use broadcasting to set the mask values for the specified channels
    final_mask[:, channels, :, :] = mask.view(B, len(channels), H, W)

    return final_mask

def create_patch_mask(tensor, channels=None, ratio=0.1):
    B, C, H, W = tensor.shape
    if channels is None:
        channels = range(C) # Assume apply to all channels
    patch_size = int(min(H, W) * ratio)
    start = (H - patch_size) // 2
    end = start + patch_size
    mask = torch.zeros_like(tensor)
    mask[:, channels] = 1
    mask[:, channels, start:end, start:end] = 0
    return mask

def evaluate(config, epoch, pipeline, **kwargs):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        **kwargs
    ).images
    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)
    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

@torch.no_grad()
def edm_sampler(
    net, noise_scheduler, batch_size=1, class_labels=None, randn_like=torch.randn_like,
    num_inference_steps=18, S_churn=0, S_min=0, S_max=float('inf'), S_noise=0,
    deterministic=True, mask=None, same_mask=True, known_channels=None, known_latents=None,
    return_trajectory=False,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    device = 'cpu'
):
    '''
    mask: torch.Tensor, shape (H, W) or (B, C, H, W), 1 for known values, 0 for unknown
    known_latents: torch.Tensor, shape (H, W) or (B, C, H, W), known values
    '''
    if known_latents is not None:
        assert batch_size == known_latents.shape[0], "Batch size must match the known_latents shape"
        # Sample gaussian noise to begin loop

    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(net.config.sample_size, int):
        latents_shape = (
            batch_size,
            net.config.out_channels,
            net.config.sample_size,
            net.config.sample_size,
        )
    else:
        latents_shape = (batch_size, net.config.out_channels, *net.config.sample_size)

    latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=net.dtype)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    t_steps = noise_scheduler.sigmas.to(device)

    x_next = latents.to(torch.float64) * t_steps[0]
    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask[None, None, ...].expand_as(x_next)
        #if len(known_latents.shape) == 2:
        #    known_latents = known_latents[None, None, ...].expand_as(x_next)
    else:
        mask = torch.zeros_like(x_next)
    if known_latents is not None:
        x_next = x_next * (1 - mask) + known_latents * mask

    if return_trajectory:
        whole_trajectory = torch.zeros((num_inference_steps, *x_next.shape), dtype=torch.float64)
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        if not deterministic:
            # Increase noise temporarily.
            gamma = min(S_churn / num_inference_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur) * (1 - mask)
        else:
            t_hat = t_cur
            x_hat = x_cur

        tmp_x_hat = x_hat.clone()
        c_noise = noise_scheduler.precondition_noise(t_hat)
        # Euler step.
        if mask is not None:
            if same_mask:
                # Only use one of the known channels in this case
                concat_mask = mask[:, [known_channels[0]]]
            else:
                concat_mask = mask[:, known_channels]
            tmp_x_hat = torch.cat((tmp_x_hat, concat_mask), dim=1) # Assume unknonw values for different channels are the same

        tmp_x_hat = noise_scheduler.precondition_inputs(tmp_x_hat, t_hat)

        denoised = net(tmp_x_hat.to(torch.float32), c_noise.reshape(-1).to(torch.float32), class_labels).sample.to(torch.float64)
        denoised = noise_scheduler.precondition_outputs(x_hat, denoised, t_hat)

        d_cur = (x_hat - denoised) / t_hat # denoise has the same shape as x_hat (b, out_channels, h, w)
        x_next = x_hat + (t_next - t_hat) * d_cur * (1 - mask)

        # Apply 2nd order correction.
        if i < num_inference_steps - 1:
            tmp_x_next = x_next.clone()
            c_noise = noise_scheduler.precondition_noise(t_next)
            if mask is not None:
                tmp_x_next = torch.cat((tmp_x_next, concat_mask), dim=1) # Assume unknonw values for different channels are the same
            
            tmp_x_next = noise_scheduler.precondition_inputs(tmp_x_next, t_next)

            denoised = net(tmp_x_next.to(torch.float32),c_noise.reshape(-1).to(torch.float32), class_labels).sample.to(torch.float64)
            denoised = noise_scheduler.precondition_outputs(x_next, denoised, t_next)

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) * (1 - mask)

        if return_trajectory:
            whole_trajectory[i] = x_next

    if return_trajectory:
        return x_next, whole_trajectory
    else:
        return x_next

def ensemble_sample(pipeline, sample_size, mask, sampler_kwargs=None, class_labels=None, known_latents=None, batch_size=64,
                 sampler_type: Optional[str] = 'edm', # 'edm' or 'pipeline'
                 device='cpu',
                 ):
    batch_size_list = [batch_size]*int(sample_size/batch_size) + [sample_size % batch_size]
    #print(latents.shape, class_labels.shape, mask.shape, known_latents.shape)
    count = 0
    samples = torch.empty(sample_size, pipeline.unet.config.out_channels, pipeline.unet.config.sample_size,
                          pipeline.unet.config.sample_size, device=device, dtype=pipeline.unet.dtype)
    if sampler_kwargs is None:
        sampler_kwargs = {}
    if sampler_type == 'edm':
        model = pipeline.unet
        noise_scheduler = copy.deepcopy(pipeline.scheduler)
    with torch.no_grad():
        for num_sample in tqdm(batch_size_list):
            #tmp_class_labels = repeat(class_labels, 'C -> B C', B=num_sample)
            generator = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in range(count, count+num_sample)]
            tmp_mask = repeat(mask, '1 C H W -> B C H W', B=num_sample)
            tmp_known_latents = repeat(known_latents, '1 C H W -> B C H W', B=num_sample)
            if sampler_type == 'edm':
                tmp_samples = edm_sampler(model, noise_scheduler, batch_size=num_sample, generator=generator, device=device,
                                          class_labels=class_labels, mask=tmp_mask, known_latents=tmp_known_latents, **sampler_kwargs)
            elif sampler_type == 'pipeline':
                tmp_samples = pipeline(batch_size=num_sample, generator=generator, class_labels=class_labels,
                                        mask=tmp_mask, known_latents=tmp_known_latents, return_dict=False, **sampler_kwargs)[0]
            samples[count:count+num_sample] = tmp_samples
            count += num_sample
    return samples