import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
from typing import Union, List, Optional, Tuple
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise Exception("target not in config! ", config)
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_and_filter_config(config):
    flat_config = flatten_dict(config)
    filtered_config = {}
    for key, value in flat_config.items():
        if isinstance(value, (int, float, str, bool, torch.Tensor)):
            filtered_config[key] = value
        else:
            filtered_config[key] = str(value)  # Convert unsupported types to string
    return filtered_config

def convert_to_rgb(images):
    # Get the colormap
    cmap = plt.get_cmap('jet')
    
    # Ensure images are detached and converted to numpy for colormap application
    images_np = images.squeeze(1).detach().cpu().numpy()  # shape: (B, H, W)

    converted_images = []
    for img in images_np:
        # Normalize img to range [0, 1]
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-5)
        
        # Apply colormap and convert to RGB
        img_rgb = cmap(img_normalized)
        
        # Convert from RGBA (4 channels) to RGB (3 channels)
        img_rgb = img_rgb[..., :3]  # shape: (H, W, 3)
        
        # Convert to PyTorch tensor and scale to range [0, 255]
        img_rgb_tensor = torch.tensor(img_rgb * 255, dtype=torch.uint8).permute(2, 0, 1)  # shape: (3, H, W)
        
        converted_images.append(img_rgb_tensor)

    return converted_images

def read_hdf5_to_numpy(file_path, key, data_type=np.float32):
    with h5py.File(file_path, 'r') as f:
        #print(f'Reading dataset {key}')
        dataset = f[key]
        data_dtype = data_type
        data_array = np.asarray(dataset, dtype=data_dtype)
    return data_array

@torch.no_grad()
def calculate_covariance(samples, channel):
    """
    Calculate the covariance matrix for a selected channel in the ensemble of samples.

    Args:
        samples (torch.Tensor): The ensemble of samples with shape (B, C, H, W).
        channel (int): The index of the channel for which to calculate the covariance matrix.

    Returns:
        torch.Tensor: The covariance matrix of the selected channel with shape (H*W, H*W).
    """
    # Extract the selected channel
    selected_channel_data = samples[:, channel, :, :]  # Shape: (B, H, W)
    
    # Flatten the spatial dimensions (H, W) into a single dimension
    B, H, W = selected_channel_data.shape
    flattened_data = selected_channel_data.view(B, -1)  # Shape: (B, H*W)
    
    # Calculate the covariance matrix
    mean = torch.mean(flattened_data, dim=0, keepdim=True)  # Mean along batch dimension
    centered_data = flattened_data - mean  # Centering data
    covariance_matrix = centered_data.t() @ centered_data / (B - 1)  # Covariance matrix
    
    return covariance_matrix

def rand_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    # modified from diffusers.utils.torch_utils.randn_tensor
    """A helper function to create random tensors with uniform distribution on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slightly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.rand(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.rand(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def plot_channel(samples, channel, title, cb=False, mask=None, save=False):
    # samples need to be divisible by 4
    try:
        samples = samples.detach().cpu().numpy()
        if mask is not None:
            mask = mask.detach().cpu().numpy
    except:
        pass
    h2w_ratio = int(samples.shape[0]/4)
    fig, axes = plt.subplots(h2w_ratio, 4, figsize=(20, int(h2w_ratio*5)), sharey=True, sharex=True)
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        if i < samples.shape[0]:
            im = ax.imshow(samples[i, channel, :, :], cmap='jet')
            ax.axis('off')
            if cb:
                fig.colorbar(im, ax=ax)  # Add colorbar to the current axis
            if mask is not None:
                tmp_mask = mask[i, channel, :, :]
                mask_indices = np.where(tmp_mask == 1)
                ax.scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=7)
        else:
            ax.axis('off')  # Hide empty plots
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')

def plot_steps(samples, idx_in_batch=0, start_step=0, interval=1, mask=None):
    # samples shape: (T, B, C, H, W), mask shape: (B, C, H, W)
    try:
        samples = samples.detach().cpu().numpy()
        if mask is not None:
            mask = mask.detach().cpu().numpy()
    except:
        pass
    total_steps = samples.shape[0]  
    num_images = (total_steps - start_step) // interval 

    num_channels = samples.shape[2]
    fig, axes = plt.subplots(num_channels, num_images, figsize=(int(3*num_images), int(3*num_channels)))  # Create 2 rows of subplots

    for i in range(num_images):
        step = start_step + (i+1) * interval - 1
        for channel in range(num_channels):  
            axes[channel, i].imshow(samples[step, idx_in_batch, channel, :, :], cmap='jet')  # Plot the image at the current step
            if mask is not None:
                tmp_mask = mask[idx_in_batch, channel, :, :]  # Get the corresponding mask
                mask_indices = np.where(tmp_mask == 1)  
                axes[channel, i].scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=6)  # Overlay black crosses
            if channel == 0:
                axes[channel, i].set_title(f'Step {step+1}')  # Set the title to the current step
            axes[channel, i].axis('off')  # Hide the axis

    plt.tight_layout()
    plt.show()

def plot_one_sample(samples, num_in_batch=0, cb=True, mask=None, channel_names=None, save_name=None, dpi=300):
    '''
    samples: (B, C, H, W), mask: (B, C, H, W)
    '''
    try:
        samples = samples.detach().cpu().numpy()
        if mask is not None:
            mask = mask.detach().cpu().numpy()
    except:
        pass
    num_images = samples.shape[1]
    image = samples[num_in_batch, :, :, :]
    fig, axes = plt.subplots(num_images, 1, figsize=(4, int(4*num_images)), sharey=True, sharex=True)
    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(image[i, :, :], cmap='jet', origin='lower')
        if channel_names is not None:
            ax.set_title(channel_names[i])
        ax.axis('off')
        if cb:
            fig.colorbar(im, ax=ax)
        if mask is not None:
            tmp_mask = mask[num_in_batch, i, :, :]
            mask_indices = np.where(tmp_mask == 1)
            ax.scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=7)
    plt.tight_layout()
    #if title is not None:
    #    plt.suptitle(title, fontsize=16)
    if save_name is not None:
        plt.savefig(save_name + '.png', dpi=dpi, bbox_inches='tight')

def plot_horizontal(images_list, channel_names=None, image_names=None, save_name=None, mask=None, plot_mask_idx=[], which_cb=None, dpi=300):
    '''
    images_list: list of images, each with shape (C, H, W)
    mask: Optional, shape (C, H, W)
    which_cb: Optional, index of the image to be used for scaling the color bar
    '''
    try:
        tmp_images_list = []
        for img in images_list:
            try:
                tmp_images_list.append(img.detach().cpu().numpy())
            except:
                tmp_images_list.append(img)
        images_list = tmp_images_list
        if mask is not None:
            mask = mask.detach().cpu().numpy()
    except:
        pass
    num_images = len(images_list)
    num_channels = images_list[0].shape[0]

    fig, axes = plt.subplots(num_channels, num_images, figsize=(4*num_images, 4*num_channels), sharey=True, sharex=True)
    
    if which_cb is not None and 0 <= which_cb < num_images:
        cb_image = images_list[which_cb]
        vmin = cb_image.min(axis=(1, 2))
        vmax = cb_image.max(axis=(1, 2))
    else:
        vmin, vmax = None, None

    for img_idx, image in enumerate(images_list):
        for ch_idx in range(num_channels):
            ax = axes[ch_idx, img_idx] if num_channels > 1 else axes[img_idx]
            im = ax.imshow(image[ch_idx, :, :], cmap='jet', origin='lower', vmin=vmin[ch_idx] if vmin is not None else None, vmax=vmax[ch_idx] if vmax is not None else None)
            if mask is not None:
                if img_idx in plot_mask_idx:
                    tmp_mask = mask[ch_idx, :, :]
                    mask_indices = np.where(tmp_mask == 1)
                    ax.scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=7)
            if channel_names is not None:
                if image_names is None: 
                    ax.set_title(channel_names[ch_idx])
                else:
                    ax.set_title(f'{channel_names[ch_idx]} - {image_names[img_idx]}')
            ax.axis('off')
    
    if which_cb is not None:
        for ch_idx in range(num_channels):
            ax_pos = axes[ch_idx, -1].get_position()
            cbar_height = ax_pos.height * 0.8
            cbar_ax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0 + (ax_pos.height - cbar_height) / 2, 0.02, cbar_height])
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin[ch_idx], vmax=vmax[ch_idx]), cmap='jet'), cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save_name is not None:
        plt.savefig(save_name + '.png', dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_ensemble(samples, title, cb=True, mask=None, save=False, GT=None):
    # 1st row is mean, 2nd row is std
    # samples: (B, C, H, W), mask: (C, H, W)
    num_row = 2 if GT is None else 3
    w2h_ratio = int(samples.shape[1]/2) 
    fig, axes = plt.subplots(num_row, samples.shape[1], figsize=(int(w2h_ratio*8), 8) ,sharey=True, sharex=True)
    fig.suptitle(title, fontsize=16)
    sample_mean = torch.mean(samples, dim=0).detach().cpu().numpy()
    sample_std = torch.std(samples, dim=0).detach().cpu().numpy()
    for i, ax in enumerate(axes.flatten()):
        if i < samples.shape[1]:
            im = ax.imshow(sample_mean[i, :, :], cmap='jet')
            ax.axis('off')
            if cb:
                fig.colorbar(im, ax=ax)
            if mask is not None:
                tmp_mask = mask[i, :, :].detach().cpu().numpy()
                mask_indices = np.where(tmp_mask == 1)
                ax.scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=7)
        elif i < 2*samples.shape[1]:
            im = ax.imshow(sample_std[i-samples.shape[1], :, :], cmap='jet')
            ax.axis('off')
            if cb:
                fig.colorbar(im, ax=ax)
            if mask is not None:
                tmp_mask = mask[i-samples.shape[1], :, :].detach().cpu().numpy()
                mask_indices = np.where(tmp_mask == 1)
                ax.scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=7)
        else:
            im = ax.imshow(GT[i-2*samples.shape[1], :, :], cmap='jet')
            ax.axis('off')
            if cb:
                fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
