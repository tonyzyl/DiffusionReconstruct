import argparse
import itertools
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import UNet2DModel
import torch
import numpy as np
import pandas as pd
from dataloader.dataset_class import pdedata2dataloader
import os
import copy
from omegaconf import OmegaConf

from pipelines.pipeline_inv_prob import InverseProblem2DPipeline, InverseProblem2DCondPipeline
from models.unet2D import diffuserUNet2D
from models.unet2DCondition import diffuserUNet2DCondition, diffuserUNet2DCFG
from utils.general_utils import instantiate_from_config
from utils.vt_utils import vt_obs
from losses.metric import get_metrics_2D


logger = get_logger(__name__, log_level="INFO")

def parse_list_int(value):
    try:
        # Try to split by commas and convert to a list of integers
        steps = [int(x) for x in value.split(',')]
        return steps
    except ValueError:
        # If it fails, assume it's a single integer
        return [int(value)]
    
def parse_list_float(value):
    try:
        # Try to split by commas and convert to a list of floats
        ratios = [float(x) for x in value.split(',')]
        return ratios
    except ValueError:
        # If it fails, assume it's a single float
        return [float(value)]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Diffusers model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--repo_name', type=str, help="Repository name.")
    parser.add_argument('--subfolder', type=str, help="Subfolder in the repository.")
    parser.add_argument('--path_to_ckpt', type=str, help="Path to the checkpoint.")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument('--path_to_csv', type=str, required=True, help="Path to the CSV file for saving the results, if none exists, a new one will be created.")

    # eval
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--num_inference_steps', type=parse_list_int, help='Number of inference steps, can be a single integer or a comma-separated list of integers.')
    parser.add_argument('--mask_ratios', type=parse_list_float, help='Mask ratios, can be a single float or a comma-separated list of floats.')
    parser.add_argument('--vt_spacing', type=parse_list_int, help="Spacing for dividing domain.")
    parser.add_argument('--mode', type=str, required=True, help="Mode for evaluation, 'edm', 'pipeline', 'vt'.")
    parser.add_argument('--conditioning_type', type=str, default='xattn', help="Conditioning type for evaluation.")
    parser.add_argument('--ensemble_size', type=int, default=25, help="Ensemble size for evaluation.")
    parser.add_argument('--channel_mean', action='store_true', help="Whether to output the channel mean.")
    parser.add_argument('--structure_sampling', action='store_true', default=False, help="Whether to sample with vt.")
    parser.add_argument('--noise_level', type=float, default=0., help="Noise level for evaluation.")
    parser.add_argument('--verbose', action='store_true', help="Whether to print verbose information.")
    parser.add_argument('--total_eval', type=int, default=1000, help="Total number of evaluation samples.")

    return parser.parse_args()

def main(args):
    print(f"Received args: {args}") 
    if args.config is None:
        raise ValueError("The config argument is missing. Please provide the path to the configuration file using --config.")
    config = OmegaConf.load(args.config)

    accelerator = Accelerator()

    use_vt = False
    if 'vt' in args.config:
        use_vt = True
        assert args.mode == 'vt', 'Mode should be vt when using vt'
        model_cls = diffuserUNet2D 
        logger.info('Using vt model')
    else:
        if args.conditioning_type == 'cfg':
            model_cls = diffuserUNet2DCFG
            logger.info('Using cfg model')
        elif args.conditioning_type == 'xattn':
            model_cls = diffuserUNet2DCondition
            logger.info('Using xattn model')
        elif args.conditioning_type == 'uncond':
            model_cls = UNet2DModel
            logger.info('Using uncond model')
        else:
            raise NotImplementedError


    noise_scheduler_config = config.pop("noise_scheduler", OmegaConf.create())
    dataloader_config = config.pop("dataloader", OmegaConf.create())
    general_config = config.pop("general", OmegaConf.create())

    set_seed(general_config.seed)

    repo_name = args.repo_name
    subfolder = args.subfolder

    if args.path_to_ckpt is None:
        unet = model_cls.from_pretrained(repo_name,
                                    subfolder=subfolder,                  
                                    use_safetensors=True,
                                    )
    else:
        unet = model_cls.from_pretrained(args.path_to_ckpt,
                                    use_safetensors=True,
                                    )

    noise_scheduler = instantiate_from_config(noise_scheduler_config)

    generator = torch.Generator(device=accelerator.device).manual_seed(general_config.seed)
    _, val_dataset, _ = pdedata2dataloader(**dataloader_config, generator=generator,
                                                                return_dataset=True)

    #select_idx = np.random.choice(len(val_dataset), args.total_eval, replace=False)
    select_idx = np.arange(0, args.total_eval)
    reduced_val_dataset = torch.utils.data.Subset(val_dataset, select_idx)

    unet = accelerator.prepare_model(unet ,evaluation_mode=True)

    if args.conditioning_type == 'xattn' or args.conditioning_type == 'cfg':
        pipeline = InverseProblem2DCondPipeline(unet, scheduler=copy.deepcopy(noise_scheduler))
    elif args.conditioning_type == 'uncond':
        pipeline = InverseProblem2DPipeline(unet, scheduler=copy.deepcopy(noise_scheduler))
    else:
        raise NotImplementedError

    for num_inference_steps, mask_ratios, vt_spacing in itertools.product(args.num_inference_steps, args.mask_ratios, args.vt_spacing):
    
        sampler_kwargs = {
        "num_inference_steps": num_inference_steps,
        "known_channels": general_config.known_channels,
        #"same_mask": general_config.same_mask,
        }
        x_idx, y_idx = torch.meshgrid(torch.arange(8, 128, vt_spacing), torch.arange(8, 128, vt_spacing))
        x_idx = x_idx.flatten().to(accelerator.device)
        y_idx = y_idx.flatten().to(accelerator.device)
        if "darcy" in args.subfolder:
            mask_kwargs = {
                "x_idx": x_idx,
                "y_idx": y_idx,
                "channels": general_config.known_channels,
            }
        else:
            mask_kwargs = {
                "ratio": mask_ratios,
                "channels": general_config.known_channels,
            }

        tmp_dim = unet.config.sample_size
        vt = vt_obs(x_dim=tmp_dim, y_dim=tmp_dim, x_spacing=vt_spacing, y_spacing=vt_spacing, known_channels=general_config.known_channels, device=accelerator.device)
        if not 'ratio' in mask_kwargs:
            if 'x_idx' in mask_kwargs:
                print('Total number of observation points: ', mask_kwargs["x_idx"].shape[0], ' Perceage of known points: ', mask_kwargs["x_idx"].shape[0] / tmp_dim**2)
            else:
                print('Total number of observation points: ', vt.x_start_grid.numel(), ' Perceage of known points: ', vt.x_start_grid.numel() / tmp_dim**2)
        else:
            print('Total number of observation points: ', mask_kwargs['ratio']*tmp_dim**2, ' Perceage of known points: ', mask_kwargs['ratio'])

        err_RMSE, err_nRMSE, err_CSV = get_metrics_2D(reduced_val_dataset, pipeline=pipeline,
                                                    vt = vt,
                                                    vt_model = unet if use_vt else None,
                                                    batch_size = args.batch_size,
                                                    ensemble_size = args.ensemble_size,
                                                    sampler_kwargs = sampler_kwargs,
                                                    mask_kwargs = mask_kwargs,
                                                    known_channels=general_config.known_channels,
                                                    device = accelerator.device,
                                                    mode = args.mode, #'edm', 'pipeline', 'vt'
                                                    conditioning_type=args.conditioning_type,
                                                    inverse_transform = dataloader_config.transform, # 'normalize'
                                                    inverse_transform_args=dataloader_config.transform_args,
                                                    channel_mean=args.channel_mean,
                                                    structure_sampling=args.structure_sampling,
                                                    noise_level=args.noise_level,
                                                    verbose=args.verbose,
        )

        csv_filename = args.path_to_csv

        if args.mode != 'mean':
            if args.structure_sampling:
                index_value = f"{args.config.split('/')[-1].split('.')[0]}_spacing_{str(vt_spacing)}_mode_{args.mode}_step_{str(num_inference_steps)}"
            else:
                index_value = f"{args.config.split('/')[-1].split('.')[0]}_ratio_{str(mask_kwargs['ratio'])}_mode_{args.mode}_step_{str(num_inference_steps)}"
        else:
            index_value = f"{dataloader_config.data_name}_mean"

        if os.path.exists(csv_filename):
            df_existing = pd.read_csv(csv_filename, index_col=0)
        else:
            df_existing = pd.DataFrame(columns=['RMSE', 'nRMSE', 'CSV'])
            df_existing.index.name = 'Index'

        if "darcy" in args.subfolder:
            # For darcy we only save the permeability field
            df_new = pd.DataFrame({'RMSE': [err_RMSE.cpu().numpy()[0]],
                                'nRMSE': [err_nRMSE.cpu().numpy()[0]],
                                'CSV': [err_CSV.cpu().numpy()[0]]},
                                index=[index_value])
        else:
            df_new = pd.DataFrame({'RMSE': [err_RMSE.cpu().numpy()], 
                                'nRMSE': [err_nRMSE.cpu().numpy()],
                                'CSV': [err_CSV.cpu().numpy()]},
                                index=[index_value])

        df_combined = pd.concat([df_existing, df_new])
        df_combined.to_csv(csv_filename)

        logger.info(f'RMSE: {err_RMSE.cpu().numpy()}, nRMSE: {err_nRMSE.cpu().numpy()}, CSV: {err_CSV.cpu().numpy()}')

if __name__ == "__main__":
    args = parse_args()
    main(args)