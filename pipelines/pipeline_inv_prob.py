import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, Union, Tuple, List

from utils.pipeline_utils import Fields2DPipelineOutput

class InverseProblem2DPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, scheduler_step_kwargs: Optional[dict] = None):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)
        self.scheduler_step_kwargs = scheduler_step_kwargs or {}

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        return_dict: bool = True,
        mask: torch.Tensor = None,
        known_channels: List[int] = None,
        known_latents: torch.Tensor = None,
    ) -> Union[Fields2DPipelineOutput, Tuple]:

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.out_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.out_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        
        assert known_latents is not None, "known_latents must be provided"

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        noise = image.clone()

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            x_in = self.scheduler.scale_model_input(image, t)
            # 1. predict noise model_output
            model_output = self.unet(x_in, t, return_dict=False)[0]
            model_output = model_output * (1 - mask) + known_latents * mask

            # 2. do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image[:, :self.unet.config.out_channels], **self.scheduler_step_kwargs,
                return_dict=False
            )[0]

            tmp_known_latents = known_latents.clone()
            if i < len(self.scheduler.timesteps) - 1:
                noise_timestep = self.scheduler.timesteps[i + 1]
                tmp_known_latents = self.scheduler.add_noise(tmp_known_latents, noise, torch.tensor([noise_timestep]))

            image = image * (1 - mask) + tmp_known_latents * mask

        if not return_dict:
            return (image,)

        return Fields2DPipelineOutput(fields=image)

class InverseProblem2DCondPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, scheduler_step_kwargs: Optional[dict] = None):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)
        self.scheduler_step_kwargs = scheduler_step_kwargs or {}

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        return_dict: bool = True,
        mask: torch.Tensor = None,
        known_channels: List[int] = None,
        known_latents: torch.Tensor = None,
        add_noise_to_obs: bool = False,
    ) -> Union[Fields2DPipelineOutput, Tuple]:

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.out_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.out_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        
        assert known_latents is not None, "known_latents must be provided"

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        if add_noise_to_obs:
            noise = image.clone()
        conditioning_tensors = torch.cat((known_latents, mask[:, [known_channels[0]]]), dim=1)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            if mask is not None and not add_noise_to_obs:
                image = image * (1 - mask) + known_latents * mask
            x_in = self.scheduler.scale_model_input(image, t)
            # 1. predict noise model_output
            model_output = self.unet(x_in, t, conditioning_tensors=conditioning_tensors, return_dict=False)[0]

            # 2. do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, **self.scheduler_step_kwargs,
                return_dict=False
            )[0]
            if add_noise_to_obs:
                tmp_known_latents = known_latents.clone()
                if i < len(self.scheduler.timesteps) - 1:
                    noise_timestep = self.scheduler.timesteps[i + 1]
                    tmp_known_latents = self.scheduler.add_noise(tmp_known_latents, noise, torch.tensor([noise_timestep]))

        if mask is not None:
            image = image * (1 - mask) + known_latents * mask

        if not return_dict:
            return (image,)

        return Fields2DPipelineOutput(fields=image)