import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py

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
        dataset = f[key]
        data_dtype = data_type
        data_array = np.asarray(dataset, dtype=data_dtype)
    return data_array

def plot_channel(samples, channel, title, cb=False, mask_overlay=None, save=False):
    # samples need to be divisible by 4
    samples = samples.detach().cpu().numpy()
    h2w_ratio = int(samples.shape[0]/4)
    fig, axes = plt.subplots(h2w_ratio, 4, figsize=(20, int(h2w_ratio*5)), sharey=True, sharex=True)
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        if i < samples.shape[0]:
            im = ax.imshow(samples[i, channel, :, :], cmap='jet')
            ax.axis('off')
            if cb:
                fig.colorbar(im, ax=ax)  # Add colorbar to the current axis
            if mask_overlay is not None:
                mask = mask_overlay[i, channel, :, :].detach().cpu().numpy()
                mask_indices = np.where(mask == 1)
                ax.scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=7)
        else:
            ax.axis('off')  # Hide empty plots
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')

def plot_ensemble(samples, title, cb=True, mask_overlay=None, save=False):
    # 1st row is mean, 2nd row is std
    # samples: (B, C, H, W), mask_overlay: (C, H, W)
    w2h_ratio = int(samples.shape[1]/2) 
    fig, axes = plt.subplots(2, samples.shape[1], figsize=(int(w2h_ratio*10), 10) ,sharey=True, sharex=True)
    fig.suptitle(title, fontsize=16)
    sample_mean = torch.mean(samples, dim=0).detach().cpu().numpy()
    sample_std = torch.std(samples, dim=0).detach().cpu().numpy()
    for i, ax in enumerate(axes.flatten()):
        if i < samples.shape[1]:
            im = ax.imshow(sample_mean[i, :, :], cmap='jet')
            ax.axis('off')
            if cb:
                fig.colorbar(im, ax=ax)
            if mask_overlay is not None:
                mask = mask_overlay[i, :, :].detach().cpu().numpy()
                mask_indices = np.where(mask == 1)
                ax.scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=7)
        else:
            im = ax.imshow(sample_std[i-samples.shape[1], :, :], cmap='jet')
            ax.axis('off')
            if cb:
                fig.colorbar(im, ax=ax)
            if mask_overlay is not None:
                mask = mask_overlay[i-samples.shape[1], :, :].detach().cpu().numpy()
                mask_indices = np.where(mask == 1)
                ax.scatter(mask_indices[1], mask_indices[0], c='black', marker='x', s=7)
    plt.tight_layout()
    if save:
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
