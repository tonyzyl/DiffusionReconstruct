import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.general_utils import read_hdf5_to_numpy

def normalize_transform(sample, mean, std, target_std=1):
    # (C, H, W)
    mean = torch.tensor(mean, device=sample.device)
    std = torch.tensor(std, device=sample.device)
    return ((sample - mean[:, None, None]) / std[:, None, None]) * target_std

def inverse_normalize_transform(normalized_sample, mean, std, target_std=1):
    # (C, H, W)
    mean = torch.tensor(mean, device=normalized_sample.device)
    std = torch.tensor(std, device=normalized_sample.device)
    return (normalized_sample / target_std) * std[:, None, None] + mean[:, None, None]

class FullDataset(Dataset):
    def __init__(self, data, transform=None, transform_args=None):
        """
        Args:
            data (numpy.ndarray): Full dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform
        if transform_args is None:
            transform_args = {}
        else:
            self.transform_args = transform_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample, **self.transform_args)
        return sample

def npy2dataloader(full_dataset, batch_size, num_workers, split_ratios=(0.7, 0.2, 0.1), transform=None, transform_args=None,
                           rearrange_args=None, random_dataset=True, generator=None, return_dataset=False):
    if rearrange_args is not None:
        full_dataset = rearrange(full_dataset, rearrange_args)
    full_dataset = torch.tensor(full_dataset, dtype=torch.float32)

    train_size = int(split_ratios[0] * len(full_dataset))
    val_size = int(split_ratios[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    if random_dataset:
        print('Randomly splitting dataset.')
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    else:
        print('Splitting dataset by length.')
        train_dataset = full_dataset[:train_size]
        val_dataset = full_dataset[train_size:train_size+val_size]
        test_dataset = full_dataset[train_size+val_size:]
    
    if transform is not None:
        if transform == 'normalize':
            mean = transform_args['mean']
            std = transform_args['std']
            transform = normalize_transform
            if 'target_std' in transform_args:
                target_std = transform_args['target_std']
                transform_args = {'mean': mean, 'std': std, 'target_std': target_std}
            else:    
                transform_args = {'mean': mean, 'std': std}
        else:
            raise NotImplementedError(f'Transform: {transform} not implemented.')

    # Apply the same transform to all splits if needed
    train_dataset = FullDataset(train_dataset, transform=transform, transform_args=transform_args)
    val_dataset = FullDataset(val_dataset, transform=transform, transform_args=transform_args)
    test_dataset = FullDataset(test_dataset, transform=transform, transform_args=transform_args)

    if return_dataset:
        return train_dataset, val_dataset, test_dataset

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=generator)

    return train_loader, val_loader, test_loader

def pdedata2dataloader(data_dir, batch_size=32, num_workers=1, split_ratios=(0.7, 0.2, 0.1), transform=None, transform_args=None,
                           rearrange_args=None, generator=None, data_name=None, return_dataset=False):

    if data_name == 'darcy':
        full_dataset = np.load(data_dir)
        return npy2dataloader(full_dataset, batch_size, num_workers, split_ratios=split_ratios, transform=transform, transform_args=transform_args,
                              rearrange_args=rearrange_args, random_dataset=False, generator=generator, return_dataset=return_dataset)
    elif data_name == 'shallow_water':
        full_dataset = np.load(data_dir)
        return npy2dataloader(full_dataset, batch_size, num_workers, split_ratios=split_ratios, transform=transform, transform_args=transform_args,
                              rearrange_args=rearrange_args, random_dataset=False, generator=generator, return_dataset=return_dataset)
    elif data_name == 'compressible_NS':
        keys = ['Vx' , 'Vy', 'density', 'pressure']
        for count, key in enumerate(keys):
            data = read_hdf5_to_numpy(data_dir, key)
            data = rearrange(data, 'n t x y -> (n t) x y')
            if count == 0:
                data_len = data.shape[0]
                full_dataset = np.empty((data_len, len(keys), data.shape[1], data.shape[2]), dtype=np.float32)
            full_dataset[:, count, :, :] = data
        return npy2dataloader(full_dataset, batch_size, num_workers, split_ratios=split_ratios, transform=transform, transform_args=transform_args,
                              rearrange_args=rearrange_args, random_dataset=False, generator=generator, return_dataset=return_dataset)
    elif data_name == 'diffusion_reaction':
        keys = ['data']
        full_dataset = read_hdf5_to_numpy(data_dir, keys[0])
        full_dataset = rearrange(full_dataset, 'n t x y c -> (n t) c x y')
        return npy2dataloader(full_dataset, batch_size, num_workers, split_ratios=split_ratios, transform=transform, transform_args=transform_args,
                              rearrange_args=rearrange_args, random_dataset=False, generator=generator, return_dataset=return_dataset)
    else:
        raise NotImplementedError(f'Dataset: {data_name} not implemented.')