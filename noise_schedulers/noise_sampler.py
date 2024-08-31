import torch

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