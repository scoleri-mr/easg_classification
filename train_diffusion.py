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
import torch.nn as nn
from torch_geometric.data import Data
import wandb

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import EASGvae, EASGAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils_diffusion import create_dataset, CustomDataset, linear_beta_schedule, read_stats, eval_autoencoder, construct_nx_from_adj, store_stats, gen_stats, calculate_mean_std, evaluation_metrics, z_score_norm
from dataset_ae import EASGDatasetAE
from utils import load_model, save_checkpoint

from torch.utils.data import Subset
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
np.random.seed(13)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='TrainDiffusion')
    parser.add_argument('--ann_path', type=str, default='./annts_in_new_format/', help='path to annotations')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to ROI and clip features')
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--n_max_nodes', type=int, default=100)
    parser.add_argument('--spectral_emb_dim', type=int, default=10)
    parser.add_argument('--epochs_denoise', type=int, default=2000)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--hidden_dim_denoise', type=int, default=256)
    parser.add_argument('--n_layers_denoise', type=int, default=3)
    parser.add_argument('--no_train_denoiser', action='store_false', dest='train_denoiser', help="If specified, do not train the denoiser.")
    parser.add_argument('--n_properties', type=int, default=0)
    parser.add_argument('--dim_condition', type=int, default=0)
    parser.add_argument('--cond', action='store_true', help='If specified use conditional generation, otherwise conditioning is switched off.')
    parser.add_argument('--no_wandb', action='store_false', dest='wandb', help="If specified disables wandb logging")
    parser.add_argument('--wandb_proj', type=str, default='diffusion_recon')
    parser.add_argument('--evaluation', action='store_true', help='Evaluation mode')
    parser.add_argument('--diffusion_path', type=str, help='path to the trained diffusion model')
    parser.add_argument('--vae_path', type=str, help='path to the trained vae', default='experiments/best_VAE1000_sep=True_od=256_kld=original_b=0.0005_lr=0.0001_fl=True_ex=False_eps=0.1_1719244127/checkpoints/last.ckpt')
    parser.add_argument('--norm_type', type=str, help='normalization layer for diffusion model', default='layer')
    parser.add_argument('--scheduler_type', type=str, help="Choose 'step' for StepLR and 'warmup' for CosineAnnealingWarmupRestarts. Use lr as parameter for max_lr in warmup.", default='warmup')
    parser.add_argument('--min_lr', type=float, help="Minimum learning rate for CosineAnnealingWarmupRestarts", default=0.00001)
    parser.add_argument('--loss_type', type=str, help="Choose between 'huber' and 'l2'", default='huber')
    parser.add_argument('--train_mode', type=str, default='noise', help="Select diffusion objective: 'reconstruct' to predict reconstructed samples, 'noise' to predict noise.")
    args = parser.parse_args()
    return args

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get datasets and split in train, test and validation
    args = parse_args()
    with open(args.ann_path + 'verbs.txt') as f:
        verbs = [l.strip() for l in f.readlines()]
    num_verbs = len(verbs)

    with open(args.ann_path + 'objects.txt') as f:
        objs = [l.strip() for l in f.readlines()]
    num_objs = len(objs)

    with open(args.ann_path + 'relationships.txt') as f:
        rels = [l.strip() for l in f.readlines()]
    num_rels = len(rels)

    path_annts = Path(args.ann_path)
    path_data = Path(args.data_path)

    # check if conditioning is activated
    if args.cond:
        print('Conditioning applied.')
        n_properties = args.n_properties
        dim_condition = args.dim_condition
    else:
        print('No conditioning applied.')
        n_properties = 0
        dim_condition = 0

    # check train mode:
    if args.train_mode == 'noise':
        print("Training with noise as objective function.")
    elif args.train_mode == 'reconstruct':
        print("Training with denoised sample as objective function.")
    else: 
        raise Exception("Wrong taining mode.")

    # original dataset only has train and validation, 
    validation_dataset = EASGDatasetAE(path_annts, path_data, 'val', verbs, objs, rels)
    train_dataset = EASGDatasetAE(path_annts, path_data, 'train', verbs, objs, rels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # load the variational autoencoder
    vae = load_model('vae', args.vae_path, separate=True, output_dim=args.latent_dim)
    vae = vae.to(device)
    vae.eval()

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

    denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=n_properties, d_cond=dim_condition, norm_type=args.norm_type).to(device)
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
            args.exp_name = f"diffusion_tsteps={args.timesteps}_nlayer={args.n_layers_denoise}_ldim={args.latent_dim}_lr={args.lr}_sch={args.scheduler_type}_loss={args.loss_type}_dcond={args.dim_condition}_mode={args.train_mode}_{str(int(time.time()))}"

        if args.wandb:
            wandb.init(project=f'{args.wandb_proj}', config=args, name=args.exp_name)
            wandb.watch(denoise_model, log="all")   

        # Train denoising model
        best_val_loss = np.inf
        for epoch in range(1, args.epochs_denoise+1):
            denoise_model.train()
            train_loss_all = 0
            train_count = 0
            for data in train_loader:
                batch, verb_gt, rel_gt = data
                batch = batch.to(device)
                with torch.no_grad():
                    x_g = vae.encode(batch)
                if args.cond:
                    conditioning = batch.stats
                else: conditioning = None
                optimizer.zero_grad()
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                loss = p_losses(denoise_model, x_g, t, conditioning, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=args.loss_type, mode=args.train_mode)
                loss.backward()
                train_loss_all += x_g.size(0) * loss.item()
                train_count += x_g.size(0)
                optimizer.step()
                if args.scheduler_type=='warmup':
                    scheduler.step()
                    if args.wandb: wandb.log({"learning_rate": scheduler.get_lr()[0]})

            if epoch % 5 == 0 or epoch == args.epochs_denoise:
                # call the evaluation
                denoise_model.eval()
                val_loss_all = 0
                val_count = 0
                for data in val_loader:
                    batch, verb_gt, rel_gt = data
                    batch = batch.to(device)
                    with torch.no_grad():
                        x_g = vae.encode(batch)
                    if args.cond:
                        conditioning = batch.stats
                    else: conditioning = None
                    t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                    loss = p_losses(denoise_model, x_g, t, conditioning, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=args.loss_type)
                    val_loss_all += x_g.size(0) * loss.item()
                    val_count += x_g.size(0)

                # log info 
                dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))
                if args.wandb: wandb.log({"train loss": train_loss_all/train_count})
                if args.wandb: wandb.log({"val loss": val_loss_all/val_count})

                # checkpoint
                save_dir = f"./experiments/diffusion_recon/{args.exp_name}/checkpoints"
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