import torch
import numpy as np
from einops import rearrange

class Karras_sigmas_lognormal:
    def __init__(self, sigmas, P_mean=-1.2, P_std=1.2, sigma_min=0.002, sigma_max=80, num_train_timesteps=1000):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = sigmas

    def __call__(self, batch_size, generator=None, device='cpu'):
        rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device, generator=generator)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Find the indices of the closest matches
        # Expand self.sigmas to match the batch size
        # sigmas get concatenated with 0 at the end
        sigmas_expanded = self.sigmas[:-1].view(1, -1).to(device)
        sigma_expanded = sigma.view(batch_size, 1)

        # Calculate the difference and find the indices of the minimum difference
        diff = torch.abs(sigmas_expanded - sigma_expanded)
        indices = torch.argmin(diff, dim=1)

        return indices

def edm_sampler(
    net, noise_scheduler, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0,
    deterministic=False, mask=None, known_latents=None,
    mask_channel=False, input_known_channel=None, res_channel=False, res_func=None
):
    '''
    mask: torch.Tensor, shape (H, W) or (B, C, H, W), 1 for known values, 0 for unknown
    known_latents: torch.Tensor, shape (H, W) or (B, C, H, W), known values
    '''
    noise_scheduler.set_timesteps(num_steps, device=latents.device)
    # Adjust noise levels based on what's supported by the network.
    
    # Time step discretization.
    #step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    #t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
    #            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    #t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
    #print(t_steps)
    t_steps = noise_scheduler.sigmas.to(latents.device)
    #print(noise_scheduler.sigmas)

    x_next = latents.to(torch.float64) * t_steps[0]
    if mask is not None:
        # mask out the known values
        assert(mask.shape[-2:] == x_next.shape[-2:]) # match (H, W)
        assert(known_latents[-2:] == x_next.shape[-2:], "Shape: ", known_latents.shape, x_next.shape)
        if len(mask.shape) == 2:
            mask = mask[None, None, ...].expand_as(x_next)
        #if len(known_latents.shape) == 2:
        #    known_latents = known_latents[None, None, ...].expand_as(x_next)
        if not mask_channel:
            Warning("mask_channel is False, but mask is not None")
    else:
        mask = torch.zeros_like(x_next)
    if known_latents is not None:
        x_next = x_next * (1 - mask) + known_latents * mask
    if res_channel:
        assert res_func is not None, "res_func is required for res_channel"

    whole_trajectory = torch.zeros((num_steps, *x_next.shape), dtype=torch.float64)
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        if not deterministic:
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur) * (1 - mask)
        else:
            t_hat = t_cur
            x_hat = x_cur

        tmp_x_hat = x_hat.clone()
        c_noise = noise_scheduler.precondition_noise(t_hat)
        tmp_x_hat = noise_scheduler.precondition_inputs(tmp_x_hat, c_noise)
        # Euler step.
        '''
        if not mask_channel:
            denoised = net(x_hat, repeat(t_hat.reshape(-1), 'w -> h w', h=x_hat.shape[0]), class_labels).to(torch.float64)
        else:
            denoised = net(torch.cat((x_hat, mask[:, :1]), dim=1), repeat(t_hat.reshape(-1), 'w -> h w', h=x_hat.shape[0]), class_labels).to(torch.float64)
        '''
        
        if mask_channel:
            tmp_x_hat = torch.cat((tmp_x_hat, mask[:, input_known_channel]), dim=1) # Assume unknonw values for different channels are the same
        if res_channel:
            tmp_x_hat = torch.cat((tmp_x_hat, rearrange(res_func(tmp_x_hat), 'b h w -> b 1 h w')), dim=1)
        denoised = net(tmp_x_hat.to(torch.float32), c_noise.reshape(-1).to(torch.float32), class_labels).sample.to(torch.float64)
        denoised = noise_scheduler.precondition_outputs(x_hat, denoised, t_hat)

        d_cur = (x_hat - denoised) / t_hat # denoise has the same shape as x_hat (b, out_channels, h, w)
        x_next = x_hat + (t_next - t_hat) * d_cur * (1 - mask)

        # Apply 2nd order correction.
        if i < num_steps - 1:
            '''
            if not mask_channel:
                denoised = net(x_next, repeat(t_next.reshape(-1), 'w -> h w', h=x_next.shape[0]), class_labels).to(torch.float64)
            else:
                denoised = net(torch.cat((x_next, mask[:, :1]), dim=1), repeat(t_next.reshape(-1), 'w -> h w', h=x_next.shape[0]), class_labels).to(torch.float64)
            '''
            tmp_x_next = x_next.clone()
            c_noise = noise_scheduler.precondition_noise(t_next)
            tmp_x_next = noise_scheduler.precondition_inputs(tmp_x_next, c_noise)
            if mask_channel:
                tmp_x_next = torch.cat((tmp_x_next, mask[:, input_known_channel]), dim=1) # Assume unknonw values for different channels are the same
            if res_channel:
                tmp_x_next = torch.cat((tmp_x_next, rearrange(res_func(tmp_x_next), 'b h w -> b 1 h w')), dim=1)
            denoised = net(tmp_x_next.to(torch.float32),c_noise.reshape(-1).to(torch.float32), class_labels).sample.to(torch.float64)
            denoised = noise_scheduler.precondition_outputs(x_next, denoised, t_next)

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) * (1 - mask)

        whole_trajectory[i] = x_next

    return x_next, whole_trajectory