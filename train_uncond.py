import argparse
import logging
import os
from datetime import timedelta
import torch
import shutil
from packaging import version
import accelerate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, InitProcessGroupKwargs
from accelerate.logging import get_logger
from diffusers import UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from torch.optim import AdamW
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from pathlib import Path
import math
from huggingface_hub import upload_folder

from noise_schedulers.noise_sampler import Karras_sigmas_lognormal
from utils.general_utils import instantiate_from_config, flatten_and_filter_config, convert_to_rgb
from utils.inverse_utils import create_scatter_mask
from utils.pipeline_utils import get_sigmas
from dataloader.dataset_class import pdedata2dataloader
from pipelines.pipeline_inv_prob import InverseProblem2DPipeline
from losses.metric import metric_func_2D

logger = get_logger(__name__, log_level="INFO")

@torch.no_grad()
def log_validation(phase_name, config, epoch, unet, noise_scheduler_config, accelerator, known_latents=None):
    # Generate some sample images
    logger.info(f"Running evaluation on {phase_name} set")
    noise_scheduler = instantiate_from_config(noise_scheduler_config)
    pipeline = InverseProblem2DPipeline(unet, scheduler=noise_scheduler)
    pipeline.to(accelerator.device)
    image_dim = pipeline.unet.config.sample_size
    generator = torch.Generator(device='cpu').manual_seed(config.seed) # Use a separate torch generator to avoid rewinding the random state of the main training loop
    tmp_latents = known_latents[:config.eval_batch_size]
    mask = create_scatter_mask(tmp_latents, channels=config.known_channels, ratio=0.015625,generator=generator, device=accelerator.device)
    #'''
    sample_images = pipeline(
        batch_size=config.eval_batch_size,
        generator=generator,
        mask=mask,
        #same_mask=config.same_mask,
        known_channels=config.known_channels,
        known_latents=tmp_latents,
        num_inference_steps = 20,
        return_dict=False,
    )[0]
    try:
        channel_names = config.channel_names
    except:
        channel_names = ['' for _ in range(sample_images.shape[1])]

    #pressure = convert_to_rgb(sample_images[:, 0].reshape(-1, 1, 64, 64))
    #permeability = convert_to_rgb(sample_images[:, 1].reshape(-1, 1, 64, 64))
    images_list = []
    GT_list = []
    for i in range(sample_images.shape[1]):
        tmp_image = convert_to_rgb(sample_images[:, i].reshape(-1, 1, image_dim, image_dim))
        ground_truth = convert_to_rgb(known_latents[:config.eval_batch_size, i].reshape(-1, 1, image_dim, image_dim))
        images_list.append(make_grid(torch.stack(tmp_image)))
        GT_list.append(make_grid(torch.stack(ground_truth)))

    err_RMSE, err_nRMSE, err_CSV = metric_func_2D(sample_images, known_latents[:config.eval_batch_size], mask=mask)
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            for i, (img, gt) in enumerate(zip(images_list, GT_list)):
                tracker.writer.add_image(phase_name + ' sample ' + channel_names[i], img, epoch)
                tracker.writer.add_image(phase_name + ' GT ' + channel_names[i], gt, epoch)
            
            accelerator.log({"RMSE": err_RMSE, "nRMSE": err_nRMSE, "CSV": err_CSV}, step=epoch)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Diffusers model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")

    return parser.parse_args()

def main(args):
    # modified based on: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py

    config = OmegaConf.load(args.config)
    tracker_config = flatten_and_filter_config(OmegaConf.to_container(config, resolve=True))

    unet_config = OmegaConf.to_container(config.pop("unet", OmegaConf.create()), resolve=True)
    noise_scheduler_config = config.pop("noise_scheduler", OmegaConf.create())
    accelerator_config = config.pop("accelerator", OmegaConf.create())
    loss_fn_config = config.pop("loss_fn", OmegaConf.create())
    optimizer_config = config.pop("optimizer", OmegaConf.create())
    lr_scheduler_config = config.pop("lr_scheduler", OmegaConf.create())    
    dataloader_config = config.pop("dataloader", OmegaConf.create())
    ema_config = config.pop("ema", OmegaConf.create())
    general_config = config.pop("general", OmegaConf.create())

    if general_config.do_edm_style_training != ('EDM' in noise_scheduler_config.target):
        raise ValueError("EDM style training is only supported for EDM noise schedulers.")
    
    if general_config.do_edm_style_training and general_config.snr_gamma is not None:
        raise ValueError("Min-SNR formulation is not supported when conducting EDM-style training.")

    set_seed(general_config.seed)

    unet = UNet2DModel.from_config(config=unet_config)

    logging_dir = Path(general_config.output_dir, general_config.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=general_config.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator = Accelerator(
        project_config=accelerator_project_config,
        **accelerator_config,
        kwargs_handlers=[kwargs]
    )

    # Create EMA for the model. Some examples:
    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
    # https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
    if ema_config.use_ema:
        ema_model = EMAModel(
            unet.parameters(),
            decay=ema_config.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=ema_config.ema_inv_gamma,
            power=ema_config.ema_power,
            model_cls=UNet2DModel,
            model_config=unet.config,
            foreach = ema_config.foreach,
        )

    # Does not work with torch.compile()
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if ema_config.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if ema_config.use_ema:
                # TODO: follow up on loading checkpoint with EMA, ema_kwargs not properly loaded
                # https://github.com/huggingface/diffusers/discussions/8802
                load_model = EMAModel.from_pretrained(
                    #os.path.join(input_dir, "unet_ema"), UNet2DModel
                    os.path.join(input_dir, "unet_ema"), UNet2DModel, #foreach=ema_config.foreach # not yet released in v0.29.2
                )
                ema_model.load_state_dict(load_model.state_dict())
                if ema_config.offload_ema:
                    ema_model.pin_memory()
                else:
                    ema_model.to(accelerator.device) 
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    noise_scheduler = instantiate_from_config(noise_scheduler_config)
    #inv_noise_scheduler_class = get_inv_noise_scheduler(noise_scheduler_config["scheduler_name"]) 
    #noise_scheduler = inv_noise_scheduler_class(**noise_scheduler_config["scheduler_params"])

    if general_config.do_edm_style_training:
        noise_sampler = Karras_sigmas_lognormal(noise_scheduler.sigmas, P_mean=1.2, P_std=1.7)

    #loss_fn = instantiate_from_config(loss_fn_config)

    generator = torch.Generator(device='cpu').manual_seed(general_config.seed)
    with accelerator.main_process_first():
        # https://github.com/huggingface/accelerate/issues/503
        # https://discuss.huggingface.co/t/shared-memory-in-accelerate/28619
        train_dataloader, val_dataloader, _ = pdedata2dataloader(**dataloader_config, generator=generator)
    val_dataset_len = len(val_dataloader.dataset)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if general_config.scale_lr:
        optimizer_config.lr = (
            optimizer_config.lr
            * accelerator.num_processes
            * accelerator.gradient_accumulation_steps
            * dataloader_config.batch_size
        )

    optimizer = AdamW(unet.parameters(), **optimizer_config)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    if "num_training_steps" not in general_config:
        general_config.num_training_steps = num_update_steps_per_epoch * general_config.num_epochs
        logger.info(f"num_training_steps not found in lr_scheduler_config. Setting num_training_steps to product of num_epochs and training dataloader length: {general_config.num_training_steps}")
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(lr_scheduler_config.name, optimizer, 
                                 num_warmup_steps = lr_scheduler_config.num_warmup_steps * accelerator.num_processes,
                                 num_training_steps = general_config.num_training_steps * accelerator.num_processes,
                                 num_cycles = lr_scheduler_config.num_cycles,
                                 power = lr_scheduler_config.power)

    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if ema_config.use_ema:
        if ema_config.offload_ema:
            ema_model.pin_memory()
        else:
            ema_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    if overrode_max_train_steps:
        general_config.num_training_steps = general_config.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    general_config.num_epochs = math.ceil(general_config.num_training_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        print(tracker_config)
        accelerator.init_trackers(general_config.tracker_project_name, config=tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        # https://github.com/huggingface/diffusers/issues/6503
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    total_batch_size = dataloader_config.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps

    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {general_config.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {dataloader_config.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {general_config.num_training_steps}")
    logger.info(f"  Total training epochs = {general_config.num_epochs}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(general_config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if accelerator.is_main_process: # temp fix for only having one random state
                accelerator.load_state(os.path.join(general_config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, general_config.num_training_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Now you train the model
    for epoch in range(first_epoch, general_config.num_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                clean_images = batch
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                # diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
                if not general_config.do_edm_style_training:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                    )
                    timesteps = timesteps.long()
                else:
                    # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
                    # instead of discrete timesteps, so here we sample indices to get the noise levels
                    # from `scheduler.timesteps`
                    # The scheduler init and step has: self.timesteps = self.precondition_noise(sigmas)
                    #indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,))
                    indices = noise_sampler(bs, device='cpu')
                    timesteps = noise_scheduler.timesteps[indices].to(device=clean_images.device)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                #mask = create_scatter_mask(clean_images, channels=general_config.known_channels, ratio=torch.rand(bs, device=clean_images.device)*0.1)
                #noise = noise * (1 - mask)
                """
                if general_config.same_mask:
                    # Only use one of the known channels in this case
                    concat_mask = mask[:, [general_config.known_channels[0]]]
                else:
                    concat_mask = mask[:, general_config.known_channels]
                """
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                #noisy_images = torch.concatenate((noisy_images, concat_mask), dim=1)

                if general_config.do_edm_style_training:
                    sigmas = get_sigmas(noise_scheduler, timesteps, len(noisy_images.shape), noisy_images.dtype, device=accelerator.device)
                    if "EDM" in noise_scheduler_config.target:
                        x_in = noise_scheduler.precondition_inputs(noisy_images, sigmas) #scale_model_input designed for step
                    else:
                        x_in = noisy_images / ((sigmas**2 + 1) ** 0.5) # By default, std=1 for other noise schedulers
                #x_in = torch.concatenate((x, concat_mask), dim=1)
                model_pred = unet(x_in if general_config.do_edm_style_training else noisy_images,
                                    timesteps, 
                                    return_dict=False)[0]
                weighting = None
                if general_config.do_edm_style_training:
                    # Similar to the input preconditioning, the model predictions are also preconditioned
                    # on noised model inputs (before preconditioning) and the sigmas.
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    if "EDM" in noise_scheduler_config.target:
                        model_pred = noise_scheduler.precondition_outputs(noisy_images[:, :unet_config['out_channels']], model_pred, sigmas) # the last (or more) channel is the mask
                        weighting = (sigmas ** 2 + 0.5** 2) / (sigmas * 0.5) ** 2 # assume sigma_data=0.5 for now
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_images
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                noisy_images / (sigmas**2 + 1)
                            )                        
                    # (comment from diffuser) We are not doing weighting here because it tends result in numerical problems.
                    # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                    # There might be other alternatives for weighting as well:
                    # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                    if "EDM" not in noise_scheduler_config.target:
                        weighting = (sigmas**-2.0).float()
                    #loss = (weighting.float() * ((clean_images.float() - model_output.float()) ** 2)).mean()
                    #loss = ((clean_images.float() - model_output.float()) ** 2).mean()
                    #loss = loss_fn(model_output, clean_images, sigmas)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = clean_images if general_config.do_edm_style_training else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        clean_images
                        if general_config.do_edm_style_training
                        else noise_scheduler.get_velocity(clean_images, noise, timesteps)
                    )
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if general_config.snr_gamma is None:
                    if weighting is not None:
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                                target.shape[0], -1
                            ),
                            1,
                        )
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, general_config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                train_loss += loss.item() / accelerator.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if ema_config.use_ema:
                    if ema_config.offload_ema:
                        ema_model.to(device="cuda", non_blocking=True)
                    ema_model.step(unet.parameters())
                    if ema_config.offload_ema:
                        ema_model.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                logs = {"train loss": train_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                if ema_config.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                global_step += 1
                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % general_config.checkpointing_steps == 0:

                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(general_config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(general_config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(general_config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if ema_config.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)

            if global_step >= general_config.num_training_steps:
                break

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:

            if (epoch + 1) % general_config.save_image_epochs == 0 or epoch == general_config.num_epochs - 1:
                if ema_config.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                select_idx = torch.randint(0, val_dataset_len, (general_config.eval_batch_size,))
                tmp_batch = torch.stack([val_dataloader.dataset[i] for i in select_idx], dim=0).to(accelerator.device) 
                log_validation('val', general_config, epoch, unwrap_model(unet), noise_scheduler_config, accelerator=accelerator,
                                known_latents=tmp_batch)
                if ema_config.use_ema:
                    # Restore the UNet parameters.
                    ema_model.restore(unet.parameters())


            if (epoch + 1) % general_config.save_model_epochs == 0 or epoch == general_config.num_epochs - 1:
                # save the model
                logger.info(f"Saving model at epoch {epoch+1}")
                if ema_config.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                unwrap_model(unet).save_pretrained(os.path.join(general_config.output_dir, "unet"))

                if ema_config.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=args.hub_model_id,
                        folder_path=general_config.output_dir+"/unet",
                        path_in_repo=general_config.output_dir.split("/")[-1],
                        commit_message="running weight",
                        ignore_patterns=["checkpoint_"],
                        token=args.hub_token if args.hub_token else None,
                    )

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)