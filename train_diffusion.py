import argparse
import os
import random
import scipy as sp
from pathlib import Path

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import AutoEncoder, VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils_diffusion import create_dataset, CustomDataset, linear_beta_schedule, read_stats, eval_autoencoder, construct_nx_from_adj, store_stats, gen_stats, calculate_mean_std, evaluation_metrics, z_score_norm
from dataset_ae import EASGDatasetAE
from utils import load_model

from torch.utils.data import Subset
np.random.seed(13)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='NeuralGraphGenerator')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--n-max-nodes', type=int, default=100)
    parser.add_argument('--spectral-emb-dim', type=int, default=10)
    parser.add_argument('--epochs-denoise', type=int, default=100)
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--hidden-dim-denoise', type=int, default=512)
    parser.add_argument('--n-layers_denoise', type=int, default=3)
    parser.add_argument('--train-denoiser', action='store_true', default=True)
    parser.add_argument('--n-properties', type=int, default=15)
    parser.add_argument('--vae_path', type=str, help='path to the trained vae')
    parser.add_argument('--dim-condition', type=int, default=128)
    parser.add_argument('--cond', action='store true', help='If specified use conditional generation, otherwise conditioning is switched off.')
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

    # original dataset only has train and validation, 
    # further splitting the original 'train' dataset in train and test
    validation_dataset = EASGDatasetAE(path_annts, path_data, 'val', verbs, objs, rels)
    dataset = EASGDatasetAE(path_annts, path_data, 'train', verbs, objs, rels)
    
    indices = torch.randperm(len(dataset)).tolist()
    train_len = int(0.8 * len(dataset))
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)
    
    # load the variational autoencoder
    vae = load_model('vae', args.vae_path, separate=True)

    # TODO: edit from here on out

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

    denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_properties, d_cond=args.dim_condition).to(device)
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    trainable_params_diff = sum(p.numel() for p in denoise_model.parameters() if p.requires_grad)
    print("Number of Diffusion model's trainable parameters: "+str(trainable_params_diff))

    if args.train_denoiser:
        # Train denoising model
        best_val_loss = np.inf
        for epoch in range(1, args.epochs_denoise+1):
            denoise_model.train()
            train_loss_all = 0
            train_count = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                x_g = vae.encode(data)
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
                loss.backward()
                train_loss_all += x_g.size(0) * loss.item()
                train_count += x_g.size(0)
                optimizer.step()

            denoise_model.eval()
            val_loss_all = 0
            val_count = 0
            for data in val_loader:
                data = data.to(device)
                x_g = vae.encode(data)
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
                val_loss_all += x_g.size(0) * loss.item()
                val_count += x_g.size(0)

            if epoch % 5 == 0:
                dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

            scheduler.step()

            if best_val_loss >= val_loss_all:
                best_val_loss = val_loss_all
                torch.save({
                    'state_dict': denoise_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, 'denoise_model.pth.tar')
    else:
        checkpoint = torch.load('denoise_model.pth.tar')
        denoise_model.load_state_dict(checkpoint['state_dict'])

    denoise_model.eval()

    del train_loader, val_loader


    ground_truth = []
    pred = []


    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        data = data.to(device)
        stat = data.stats
        bs = stat.size(0)
        samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample)
        stat_d = torch.reshape(stat, (-1, args.n_properties))

        for i in range(stat.size(0)):
            #adj = autoencoder.decode_mu(samples[random_index])
            # Gs_generated.append(construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy()))
            stat_x = stat_d[i]

            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
            stat_x = stat_x.detach().cpu().numpy()
            ground_truth.append(stat_x)
            pred.append(gen_stats(Gs_generated))


    store_stats(ground_truth, pred, "y_stats.txt", "y_pred_stats.txt")

    # stats = torch.cat(stats, dim=0).detach().cpu().numpy()


    mean, std = calculate_mean_std(ground_truth)


    mse, mae, norm_error = evaluation_metrics(ground_truth, y_pred)


    mse_all, mae_all, norm_error_all, mean_perc_error_all = z_score_norm(ground_truth, pred, mean, std)



    feats_lst = ["number of nodes", "number of edges", "density","max degree", "min degree", "avg degree","assortativity","triangles","avg triangles","max triangles","avg clustering coef", "global clustering coeff", "max k-core", "communities","diameter"]
    id2feats = {i:feats_lst[i] for i in range(len(mse))}




    print("MSE for the samples in all features is equal to: "+str(mse_all))
    print("MAE for the samples in all features is equal to: "+str(mae_all))
    print("Symmetric Mean absolute Percentage Error for the samples for all features is equal to: "+str(norm_error_all*100))
    print("=" * 100)

    for i in range(len(mse)):
        print("MSE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mse[i]))
        print("MAE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mae[i]))
        print("Symmetric Mean absolute Percentage Error for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(norm_error[i]*100))
        print("=" * 100)
