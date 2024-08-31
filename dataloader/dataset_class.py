import torch
import numpy as np
from torch.utils.data import random_split
from einops import rearrange
from torch.utils.data import Dataset, DataLoader, Subset
import xarray as xr

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

class XarrayDataset2D(Dataset):
    def __init__(
        self,
        data: xr.Dataset, # (n, t, x, y), prefferable, but have to have char 'n' and 't'
        transform: str = None,
        transform_args: dict = None,
        load_in_memory: bool = False, # TODO: figure out how to save only one copy in memory
    ):
        if not load_in_memory:
            self.data = data
        else:
            self.data = data.load()
        self.transform = self._get_transform(transform, transform_args)
        self.transform_args = transform_args or {}
        self.var_names_list = list(data.data_vars.keys())
        self.dims_dict = dict(data.dims)
        assert 'n' in self.dims_dict, 'Dataset must have dimension named "n".'
        assert 't' in self.dims_dict, 'Dataset must have dimension named "t".'
        self.length = self.dims_dict['n'] * self.dims_dict['t']

    def _get_transform(self, transform, transform_args):
        if transform == 'normalize':
            mean = transform_args['mean']
            std = transform_args['std']
            if 'target_std' in transform_args:
                target_std = transform_args['target_std']
                return lambda x: normalize_transform(x, mean, std, target_std)
            return lambda x: normalize_transform(x, mean, std)
        elif transform is None:
            return lambda x: x
        else:
            raise NotImplementedError(f'Transform: {transform} not implemented.')
    
    def _preprocess_data(self, data):
        data = torch.from_numpy(data).float()
        return self.transform(data)

    def _idx2nt(self, idx):
        return divmod(idx, self.dims_dict['t'])

    def get_array_from_xrdataset_2D(self, n_idx, t_idx):
        sliced_ds = self.data.isel({'n': n_idx, 't': t_idx})
        # xr.to_array() is slower
        return np.stack([sliced_ds[var].values for var in self.var_names_list], axis=0)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        assert idx >= 0, f'Index must be non-negative, got: {idx}.'
        n_idx, t_idx = self._idx2nt(idx)
        return self._preprocess_data(self.get_array_from_xrdataset_2D(n_idx, t_idx))

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
        train_dataset = Subset(full_dataset, list(range(0, train_size)))
        val_dataset = Subset(full_dataset, list(range(train_size, train_size+val_size)))
        test_dataset = Subset(full_dataset, list(range(train_size+val_size, len(full_dataset))))
    
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

    if not isinstance(full_dataset.data, xr.Dataset):
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

def dataset2dataloader(full_dataset, batch_size, num_workers, split_ratios=(0.7, 0.2, 0.1), random_dataset=True, generator=None, return_dataset=False):
    train_size = int(split_ratios[0] * len(full_dataset))
    val_size = int(split_ratios[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    if random_dataset:
        print('Randomly splitting dataset.')
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    else:
        print('Splitting dataset by length.')
        train_dataset = Subset(full_dataset, list(range(0, train_size)))
        val_dataset = Subset(full_dataset, list(range(train_size, train_size+val_size)))
        test_dataset = Subset(full_dataset, list(range(train_size+val_size, len(full_dataset))))

    if return_dataset:
        return train_dataset, val_dataset, test_dataset

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=generator)

    return train_loader, val_loader, test_loader

def pdedata2dataloader(data_dir, batch_size=32, num_workers=1, split_ratios=(0.7, 0.2, 0.1), transform=None, transform_args=None,
                           rearrange_args=None, generator=None, data_name=None, return_dataset=False, load_in_memory=False):

    if data_name == 'darcy':
        full_dataset = np.load(data_dir)
        return npy2dataloader(full_dataset, batch_size, num_workers, split_ratios=split_ratios, transform=transform, transform_args=transform_args,
                              rearrange_args=rearrange_args, random_dataset=False, generator=generator, return_dataset=return_dataset)
    elif data_name == 'shallow_water':
        full_dataset = np.load(data_dir)
        return npy2dataloader(full_dataset, batch_size, num_workers, split_ratios=split_ratios, transform=transform, transform_args=transform_args,
                              rearrange_args=rearrange_args, random_dataset=False, generator=generator, return_dataset=return_dataset)
    elif data_name == 'compressible_NS':
        # keys = ['Vx' , 'Vy', 'density', 'pressure']
        xarray = xr.open_dataset(data_dir, phony_dims='access', 
                                drop_variables=['t-coordinate', 'x-coordinate', 'y-coordinate'], 
                                chunks='auto')
        xarray= xarray.rename({
                                'phony_dim_0': 'n',
                                'phony_dim_1': 't',
                                'phony_dim_2': 'x',
                                'phony_dim_3': 'y'
                                })
        full_dataset = XarrayDataset2D(xarray, transform=transform, transform_args=transform_args, load_in_memory=load_in_memory)
        return dataset2dataloader(full_dataset, batch_size, num_workers, split_ratios=split_ratios, random_dataset=False, generator=generator, return_dataset=return_dataset)
    elif data_name == 'diffusion_reaction':
        # keys = ['u', 'v']
        xarray = xr.open_dataset(data_dir, phony_dims='access', 
                                drop_variables=['t-coordinate', 'x-coordinate', 'y-coordinate'], 
                                chunks='auto')
        xarray= xarray.rename({
                                'phony_dim_0': 't',
                                'phony_dim_1': 'n',
                                'phony_dim_2': 'x',
                                'phony_dim_3': 'y'
                                })
        full_dataset = XarrayDataset2D(xarray, transform=transform, transform_args=transform_args, load_in_memory=load_in_memory)
        return dataset2dataloader(full_dataset, batch_size, num_workers, split_ratios=split_ratios, random_dataset=False, generator=generator, return_dataset=return_dataset)
    else:
        raise NotImplementedError(f'Dataset: {data_name} not implemented.')