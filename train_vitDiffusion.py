import argparse
import os
import random
import scipy as sp
from pathlib import Path
import os.path as osp
import time

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import wandb

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from vitDenoise_model import DenoiseViT, p_losses, positional_encoding, sample
from utils_diffusion import linear_beta_schedule
from dataset_video.dataset_video import EASGvideo
from utils import load_model, save_checkpoint

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
np.random.seed(13)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='TrainDiffusion')
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--spectral_emb_dim', type=int, default=10)
    parser.add_argument('--epochs_denoise', type=int, default=10000)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--hidden_dim_denoise', type=int, default=256)
    parser.add_argument('--no_train_denoiser', action='store_false', dest='train_denoiser', help="If specified, do not train the denoiser.")
    parser.add_argument('--no_wandb', action='store_false', dest='wandb', help="If specified disables wandb logging")
    parser.add_argument('--wandb_proj', type=str, default='vae_objs_head')
    parser.add_argument('--evaluation', action='store_true', help='Evaluation mode')
    parser.add_argument('--diffusion_path', type=str, help='path to the trained diffusion model')
    parser.add_argument('--scheduler_type', type=str, help="Choose 'step' for StepLR and 'warmup' for CosineAnnealingWarmupRestarts. Use lr as parameter for max_lr in warmup.", default='warmup')
    parser.add_argument('--min_lr', type=float, help="Minimum learning rate for CosineAnnealingWarmupRestarts", default=0.00001)
    parser.add_argument('--loss_type', type=str, help="Choose between 'huber' and 'l2'", default='huber')
    parser.add_argument('--train_mode', type=str, default='noise', help="Select diffusion objective: 'reconstruct' to predict reconstructed samples, 'noise' to predict noise.")
    parser.add_argument('--time_dim', type=int, help="Dimention of the time positional embedding after the time mlp", default=64)
    parser.add_argument('--window_size', type=int, help="Number of frames per video", default=10)
    parser.add_argument('--threshold', type=int, help="If the number of frames per video is inferior to this treshold, the video is not considered", default=10)
    parser.add_argument('--fixed_frames', type=int, help="Number of frames that will be noise free in the diffusion model", default=5)
    parser.add_argument('--heads', type=int, help="Attention heads for the ViT", default=8)
    parser.add_argument('--depth', type=int, help="Number of attention blocks for the ViT", default=5)
    parser.add_argument('--train_path', help="path for training dataset", default="./dataset_video/encoded_videos_train_256.pth")
    parser.add_argument('--val_path', help="path for validation dataset", default="./dataset_video/encoded_videos_validation_256.pth")
    parser.add_argument('--train_triplets_path', help="path for training triplets", default="./dataset_video/train_triplets.pth")
    parser.add_argument('--val_triplets_path', help="path for validation triplets", default="./dataset_video/val_triplets.pth")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ind = args.train_path.find('.pth')
    vae_run = args.train_path[ind-4:ind]
    print(f"Reference vae run: {vae_run}")

    # get train and validation datasets
    train_dataset = EASGvideo(args.train_path, args.train_triplets_path, threshold=args.threshold, window_size=args.window_size)
    validation_dataset = EASGvideo(args.val_path, args.val_triplets_path, threshold=args.threshold, window_size=args.window_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    # check train mode:
    if args.train_mode == 'noise':
        print("Training with noise as objective function.")
    elif args.train_mode == 'reconstruct':
        print("Training with denoised sample as objective function.")
    else: 
        raise Exception("Wrong training mode.")

    # define beta schedule
    betas = linear_beta_schedule(timesteps=args.timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # creating model and optimizer
    denoise_model = DenoiseViT(depth=args.depth, heads=args.heads, d_model=args.latent_dim, hidden_dim=args.hidden_dim_denoise, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
    if args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    elif args.scheduler_type == 'warmup':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.epochs_denoise*len(train_loader), cycle_mult=1.0, max_lr=args.lr, min_lr=args.min_lr, warmup_steps=int(args.epochs_denoise*len(train_loader)/4))
    else: 
        raise Exception("Wrong scheduler type, choose between'step' and 'warmup'")

    trainable_params_diff = sum(p.numel() for p in denoise_model.parameters() if p.requires_grad)
    print("Number of Diffusion model's trainable parameters: "+str(trainable_params_diff))

    if args.train_denoiser:
        print('Training diffusion model...')
        if args.exp_name is None:
            args.exp_name = f"diffusion_{vae_run}_t={args.timesteps}_mode={args.train_mode}_lr={args.lr}_heads={args.heads}_depth={args.depth}_window={args.window_size}_fixedfr={args.fixed_frames}_treshold={args.threshold}_{str(int(time.time()))}"

        if args.wandb:
            wandb.init(project=f'{args.wandb_proj}', config=args, name=args.exp_name)
            wandb.watch(denoise_model, log="all")   

        # Train denoising model
        for epoch in range(1, args.epochs_denoise+1):
            denoise_model.train()
            train_loss_all = 0
            train_count = 0
            for batch in train_loader:
                batch = batch.to(device)
                pe = positional_encoding(args.latent_dim, args.window_size, batch.size(0))  # I need this in the loop because batch size changes at the last batch
                optimizer.zero_grad()
                t = torch.randint(0, args.timesteps, (batch.size(0),), device=device).long()
                loss = p_losses(denoise_model, batch, t, pe, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=args.loss_type, mode=args.train_mode, num_fixed_frames=args.fixed_frames)
                loss.backward()
                train_loss_all += batch.size(0) * loss.item()
                train_count += batch.size(0)
                optimizer.step()
                if args.scheduler_type=='warmup':
                    scheduler.step()
                    if args.wandb: wandb.log({"learning_rate": scheduler.get_lr()[0]})

            if epoch % 5 == 0 or epoch == args.epochs_denoise:
                # call the evaluation
                denoise_model.eval()
                val_loss_all = 0
                val_count = 0
                for batch in val_loader:
                    batch = batch.to(device)
                    pe = positional_encoding(args.latent_dim, args.window_size, batch.size(0))
                    t = torch.randint(0, args.timesteps, (batch.size(0),), device=device).long()
                    loss = p_losses(denoise_model, batch, t, pe, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=args.loss_type, num_fixed_frames=args.fixed_frames)
                    val_loss_all += batch.size(0) * loss.item()
                    val_count += batch.size(0)

                # log info 
                dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))
                if args.wandb: wandb.log({"train loss": train_loss_all/train_count})
                if args.wandb: wandb.log({"val loss": val_loss_all/val_count})

                # checkpoint
                save_dir = f"./experiments/vitDiffusion_100/{args.exp_name}/checkpoints"
                os.makedirs(save_dir, exist_ok=True)
                save_checkpoint(model=denoise_model, optimizer=optimizer, epoch=epoch, path=osp.join(save_dir, "last.ckpt"))
            
            if args.scheduler_type == 'step':
                scheduler.step()
                if args.wandb: wandb.log({"learning_rate": scheduler.get_lr()[0]})

    elif args.evaluation:
        checkpoint = torch.load(args.diffusion_path)
        denoise_model.load_state_dict(checkpoint['model_state_dict'])

if __name__ == "__main__":
    main()