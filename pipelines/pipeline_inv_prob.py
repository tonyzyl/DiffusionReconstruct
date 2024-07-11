import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, Union, Tuple, List

from utils.pipeline_utils import Fields2DPipelineOutput

class InverseProblem2DPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, scheduler_step_kwargs: Optional[dict] = None):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        #scheduler = DDIMScheduler.from_config(scheduler.config)

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
        same_mask: bool = False,
        known_channels: List[int] = None,
        known_latents: torch.Tensor = None,
        do_edm_style: bool = True,
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
        if same_mask:
            # Only use one of the known channels in this case
            concat_mask = mask[:, [known_channels[0]]]
        else:
            concat_mask = mask[:, known_channels]

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            image = image * (1 - mask) + known_latents * mask
            image = torch.concatenate((image, concat_mask), dim=1)
            # 1. predict noise model_output
            if do_edm_style:
                x_in = self.scheduler.scale_model_input(image, t)
                #x_in = torch.cat((x_in, concat_mask), dim=1)
                model_output = self.unet(x_in, t, return_dict=False)[0]
                model_output = model_output * (1 - mask) + known_latents * mask
            else:
                raise NotImplementedError("Only EDM style is supported for now")

            # 2. do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image[:, :self.unet.config.out_channels], **self.scheduler_step_kwargs,
                return_dict=False
            )[0]

        image = image * (1 - mask) + known_latents * mask

        if not return_dict:
            return (image,)

        return Fields2DPipelineOutput(fields=image)