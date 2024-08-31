import numpy as np
import torch
import matplotlib.pyplot as plt 

def get_grid_points_from_mask(batch_idx, channel_idx, mask):
    '''
    mask: torch.Tensor (B, C, H, W)
    '''
    if len(mask.shape) == 3:
        # maks (C, H, W)
        mask = mask.unsqueeze(0)
    flatten_batch_idx, flatten_channel_idx, flatten_y_idx, flatten_x_idx = torch.nonzero(mask, as_tuple=True)
    target_idx = torch.logical_and(flatten_batch_idx == batch_idx, flatten_channel_idx == channel_idx)
    return torch.column_stack((flatten_x_idx[target_idx], flatten_y_idx[target_idx]))

class vt_obs(object):
    def __init__(self, x_dim, y_dim, x_spacing, y_spacing, known_channels=None, device='cpu'):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.x_list = torch.arange(0, x_dim-x_spacing+1, x_spacing, device=device)
        self.y_list = torch.arange(0, y_dim-y_spacing+1, y_spacing, device=device)
        self.x_start_grid, self.y_start_grid = torch.meshgrid(self.x_list, self.y_list, indexing='ij')
        self.grid_x, self.grid_y = torch.meshgrid(torch.linspace(0, x_dim-1, x_dim), 
                                                  torch.linspace(0, y_dim-1, y_dim),
                                                    indexing='xy')
        self.known_channels = known_channels
        self.device = device

        self.x_start_grid = self.x_start_grid.to(device)
        self.y_start_grid = self.y_start_grid.to(device)
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)

    @torch.no_grad()
    def structure_obs(self, generator=None):
        x_offset = torch.randint(0, self.x_spacing, self.x_start_grid.shape,
                                 device=self.device, generator=generator)
        y_offset = torch.randint(0, self.y_spacing, self.y_start_grid.shape,
                                 device=self.device, generator=generator)
        x_coords = (self.x_start_grid + x_offset).flatten()
        y_coords = (self.y_start_grid + y_offset).flatten()
        return x_coords, y_coords

    @torch.no_grad()
    def _get_grid_points(self, x_coords=None, y_coords=None, generator=None):
        if x_coords is None and y_coords is None:
            x_coords, y_coords = self.structure_obs(generator=generator)
        return torch.column_stack((x_coords, y_coords)) 

    @torch.no_grad()
    def _torch_griddata_nearest(self, points, values, xi):
        distances = torch.cdist(xi, points)
        nearest_indices = torch.argmin(distances, dim=1)
        interpolated_values = values[nearest_indices]
        return interpolated_values.reshape(self.y_dim, self.x_dim)

    @torch.no_grad()
    def interpolate(self, grid_points, field):
        '''
        return griddata(
            grid_points,
            field,
            (self.grid_x, self.grid_y),
            method='nearest'
        )
        '''
        return self._torch_griddata_nearest(
            grid_points.float(), 
            field, 
            torch.stack((self.grid_x.flatten(), self.grid_y.flatten()), dim=1).float(),
        )
    
    def _plot_vt(self, known_fields, mask=None, x_coords=None, y_coords=None, plot_scatter=True, cb=True):
        '''
        known_fields: (C, H, W)
        mask: (C, H, W)
        mask_channel_idx: int, if using same mask, input the corresponding channel index
        '''
        C, H, W = known_fields.shape
        if mask is None:
            grid_points = self._get_grid_points(x_coords=x_coords, y_coords=y_coords).to(self.device)
        in_channels = C if self.known_channels is None else len(self.known_channels)
        interpolated_fields = torch.zeros(in_channels, self.y_dim, self.x_dim, dtype=known_fields.dtype)
        for idx, known_channel in enumerate(range(C) if self.known_channels is None else self.known_channels):
            if mask is not None:
                grid_points = get_grid_points_from_mask(0, known_channel, mask).to(self.device)
            field = known_fields[known_channel][grid_points[:,1], grid_points[:,0]].flatten()
            interpolated_values = self.interpolate(grid_points, field)
            interpolated_fields[idx] = torch.tensor(interpolated_values, 
                                                    dtype=known_fields.dtype, 
                                                    device=self.device)
        if x_coords is None and y_coords is None:
            x_coords, y_coords = grid_points[:,0], grid_points[:,1] 
        fig, axs = plt.subplots(in_channels, 1, figsize=(4, 4*in_channels))
        if in_channels == 1:
            axs = [axs]
        for c in range(in_channels):
            im = axs[c].imshow(interpolated_fields[c].cpu().numpy(), cmap='jet', origin='lower')
            axs[c].axis('off')
            if plot_scatter:
                axs[c].scatter(x_coords.cpu().numpy(), y_coords.cpu().numpy(), c='r', s=1)
            if cb:
                fig.colorbar(im, ax=axs[c])
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def __call__(self, known_fields, mask=None, x_coords=None, y_coords=None, generator=None):
        # known_fields: (B, C, H, W)
        B, C, _, _ = known_fields.shape
        in_channels = C if self.known_channels is None else len(self.known_channels)
        interpolated_fields = torch.zeros(B, in_channels, self.y_dim, self.x_dim, device=known_fields.device, dtype=known_fields.dtype)

        for b in range(B):
            if mask is None:
                grid_points = self._get_grid_points(x_coords=x_coords, y_coords=y_coords, generator=generator).to(self.device)

            for idx, known_channel in enumerate(range(C) if self.known_channels is None else self.known_channels):
                if mask is not None:
                    grid_points = get_grid_points_from_mask(b, known_channel, mask).to(self.device)
                field = known_fields[b, known_channel][grid_points[:,1], grid_points[:,0]].flatten()
                interpolated_values = self.interpolate(grid_points, field)
                interpolated_fields[b, idx] = interpolated_values

        return interpolated_fields