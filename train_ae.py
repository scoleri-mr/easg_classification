import os
import os.path as osp
from dataset import EASGDatasetAE
from pathlib import Path
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from utils import set_wandb_config
from models import EASG_AutoEncoder
import torch
from torch import cuda
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--ann_path', type=str,
                        default='./annts_in_new_format/', help='path to annotations')
    parser.add_argument('--data_path', type=str,
                        default='./data/', help='path to ROI and clip features')
    parser.add_argument('--num_epochs', type=int,
                        default=100, help='total number of epochs')
    parser.add_argument('--hidden_proj_dim', type=int, default=1024,
                        help='hidden dimension for linear projection')
    parser.add_argument('--proj_dim', type=int, default=512,
                        help='final dimension of verb and objects after linear projection')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='hidden dimension for the gnn')
    parser.add_argument('--output_dim', type=int, default=512,
                        help='output dimension of the gnn')
    parser.add_argument('--scheduler_type', type=str, default='cosine_annealing',
                        help='choose between step, cosine_annealing and fixed')
    parser.add_argument('--lr_start', type=float,
                        default=0.0001, help='starting learning rate')
    parser.add_argument('--lr_gamma', type=int, default=0.5,
                        help='gamma parameter for lr scheduler')
    parser.add_argument('--lr_step_size', type=int,
                        default=20, help='step size for scheduler')
    parser.add_argument('--edge_criterion', type=str, default='conc',
                        help='criterion for edge creation: elementwise mean/max or their concatenation between two adjacent nodes')
    parser.add_argument('--dropout_prob', type=float,
                        default=0.2, help='dropout probability for gnn layers')
    parser.add_argument('--wandb', dest='wandb', action='store_true')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.add_argument('--graph_type', type=str, default='gcn',
                        help='choose between graph layers: gcn, sage, gat, gin')
    parser.set_defaults(wandb=True)
    args = parser.parse_args()
    return args


def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves a checkpoint of the model and optimizer states, along with training metadata.

    Args:
    model (torch.nn.Module): The model whose parameters you want to save.
    optimizer (torch.optim.Optimizer): The optimizer with current state.
    epoch (int): Current epoch number.
    path (str): Path to save the checkpoint file.

    Returns:
    None
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')


def train(
    train_loader, validation_dataset, model, optimizer, scheduler, config, num_epochs, device, proj_dim,
    hidden_dim, output_dim, wandb_log, edge_criterion, graph_type, lr_start
):
    exp_name = f"experiments/AE_{str(int(time.time()))}"
    model = model.to(device)
    if wandb_log:
        wandb.init(project=f'easg_ae_{graph_type}', config=config)
        wandb.watch(model, log="all")

    for epoch in range(num_epochs):
        model.train()
        count = 0
        for bidx, _data in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
            batch, verb_gt, rel_gt = _data
            count += 1
            batch = batch.to(device)
            verb_gt = verb_gt.view(-1).to(device)  # [bs, ]
            # TODO: num_rels+1 is managed at the dataset level
            rel_gt = rel_gt.to(device)  # [bs, num_objs, num_rels+1]
            optimizer.zero_grad()
            out_verb, out_rel = model(batch)
            loss_verb = F.cross_entropy(input=out_verb, target=verb_gt)
            out_rel = out_rel.contiguous().view(-1, 14)
            rel_gt = rel_gt.argmax(-1).view(-1)
            loss_rel = F.cross_entropy(input=out_rel, target=rel_gt)
            loss = loss_verb + loss_rel
            loss.backward()
            optimizer.step()
            if wandb_log and bidx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({"loss": loss, "loss_verb": loss_verb,
                          "loss_rel": loss_rel, "current_lr": current_lr})
                print(
                    f"epoch {epoch}, it: {bidx}, loss: {loss.item():.4f}, loss_verb: {loss_verb.item():.4f}, loss_rel: {loss_rel.item():.4f}")
        scheduler.step()

        if epoch % 20 == 0:
            save_dir = f"./{exp_name}/checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(model=model, optimizer=optimizer,
                            epoch=epoch, path=osp.join(save_dir, "last.ckpt"))

    if wandb_log:
        wandb.finish()


def main():
    # GET TRAINING DATASET
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
    
    train_dataset = EASGDatasetAE(path_annts, path_data, 'train', verbs, objs, rels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    validation_dataset = EASGDatasetAE(path_annts, path_data, 'val', verbs, objs, rels)
    val_loader = DataLoader(validation_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)

    # DEFINE THE MODEL  AND IT'PARAMETERS
    device = 'cuda' if cuda.is_available() else print('CUDA NOT AVAILABLE')
    obj_dim = 1024  # original object dimension
    verb_dim = 2304  # original verb dimension

    if args.edge_criterion == 'mean':
        edge_criterion = 'mean'
    elif args.edge_criterion == 'max':
        edge_criterion = 'max'
    elif args.edge_criterion == 'conc':
        edge_criterion = 'conc'
    else:
        raise Exception('Wrong edge criterion')

    cosine_annealing_param = args.num_epochs

    model = EASG_AutoEncoder(obj_dim,
                             verb_dim,
                             num_rels + 1,  # TODO: added num_rels + 1
                             num_verbs, num_objs,
                             args.hidden_proj_dim,
                             args.proj_dim,
                             args.hidden_dim,
                             args.output_dim,
                             args.dropout_prob,
                             edge_criterion,
                             args.graph_type
                             )
    optimizer = Adam(model.parameters(), lr=args.lr_start)
    if args.scheduler_type == 'cosine_annealing':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_annealing_param)
    elif args.scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        raise Exception('Wrong scheduler type')

    config = set_wandb_config(args.num_epochs, args.hidden_proj_dim, args.proj_dim,
                              args.hidden_dim, args.output_dim, args.batch_size, args.scheduler_type,
                              args.lr_start, args.lr_step_size, args.lr_gamma, cosine_annealing_param,
                              args.edge_criterion, args.dropout_prob)

    # TRAIN THE MODEL
    train(train_loader, validation_dataset, model, optimizer, scheduler,
          config, args.num_epochs, device,
          args.proj_dim, args.hidden_dim, args.output_dim,
          args.wandb, args.edge_criterion, args.graph_type, args.lr_start)


if __name__ == "__main__":
    main()
