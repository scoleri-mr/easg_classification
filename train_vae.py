import os
import os.path as osp
import sys
import numpy as np
from dataset import EASGDatasetAE
from pathlib import Path
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from models import EASG_VAE
import torch
from torch import cuda
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils import *

""""
example launcher: python train_ae.py --wandb --exp_name AE_verb_rel_withVal_epochs200 --num_epochs 200
- TODO: relazioni sono sbilanciate, ce n'è una (la non presenza dell'oggetto che è predominante) - valutare alternativa
- TODO: autodecoder logic?
"""


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
    parser.add_argument('--kld_weight', type=float,
                        default=0.00025, help='beta weighting kld of vae')
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
    parser.add_argument('--wandb', action='store_true', help="If specified enables wandb logging")
    parser.add_argument('--graph_type', type=str, default='gcn',
                        help='choose between graph layers: gcn, sage, gat, gin')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint to resume')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    return args


def train(train_loader, val_loader, model, optimizer, scheduler, device, opt):
    if opt.exp_name is None:
        opt.exp_name = f"VAE_{str(int(time.time()))}"

    model = model.to(device)
    if opt.wandb:
        wandb.init(project=f'easg_ae_{opt.graph_type}', config=opt, name=opt.exp_name)
        wandb.watch(model, log="all")

    print(f"Training - exp name: {opt.exp_name}")
    for epoch in range(opt.num_epochs):
        model.train()
        for bidx, _data in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
            batch, verb_gt, rel_gt = _data
            bs = len(batch)
            batch = batch.to(device)
            verb_gt = verb_gt.view(-1).to(device)  # [bs, ]
            # TODO: num_rels+1 is managed at the dataset level
            rel_gt = rel_gt.to(device)  # [bs, num_objs, num_rels+1]
            optimizer.zero_grad()
            out_verb, out_rel, kld = model(batch)

            # compute loss
            loss_verb = F.cross_entropy(input=out_verb, target=verb_gt)
            out_rel = out_rel.contiguous().view(-1, 14)
            rel_gt = rel_gt.view(-1, 14)
            loss_rel = F.binary_cross_entropy_with_logits(input=out_rel, target=rel_gt)
            loss = loss_verb + loss_rel + (opt.kld_weight * kld)

            loss.backward()
            optimizer.step()
            if bidx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Train epoch {epoch}, it: {bidx}, loss: {loss.item():.4f}, loss_verb: {loss_verb.item():.4f}, loss_rel: {loss_rel.item():.4f}, kld: {kld.item():.4f}")
                if opt.wandb: wandb.log(
                    {"loss": loss, "loss_verb": loss_verb, "loss_rel": loss_rel, "kld": kld, "current_lr": current_lr})

        scheduler.step()

        if epoch % 10 == 0:
            ##############
            # EVALUATION #
            ##############
            model = model.eval()
            list_logits_verb, list_gt_verb = [], []
            list_logits_rel, list_gt_rel = [], []
            for bidx, _data in tqdm(enumerate(val_loader, 0), unit="batch", total=len(val_loader), desc="Validation"):
                batch, verb_gt, rel_gt = _data
                batch = batch.to(device)
                verb_gt = verb_gt.view(-1).to(device)  # [bs, ]
                # TODO: num_rels+1 is managed at the dataset level
                rel_gt = rel_gt.to(device)  # [bs, num_objs, num_rels+1]
                out_verb, out_rel, _ = model(batch)
                loss_verb = F.cross_entropy(input=out_verb, target=verb_gt)
                out_rel = out_rel.contiguous().view(-1, 14)
                # store val batch results for computing global accuracy and balanced accuracy
                list_logits_verb.append(out_verb.cpu().detach())
                list_logits_rel.append(out_rel.cpu().detach())
                list_gt_verb.append(verb_gt.cpu().detach())
                list_gt_rel.append(rel_gt.view(-1, 14).argmax(-1).view(-1).cpu().detach())

            list_logits_verb = torch.cat(list_logits_verb, dim=0)
            list_pred_verb = torch.argmax(list_logits_verb, -1)
            list_logits_rel = torch.cat(list_logits_rel, dim=0)
            list_pred_rel = torch.argmax(list_logits_rel,
                                         -1)  # TODO: this does not address multiple verb-obj relationships!
            list_gt_verb = torch.cat(list_gt_verb, dim=0)
            list_gt_rel = torch.cat(list_gt_rel, dim=0)

            # TODO: during training the accuracy of realtions is not 100% correct - we consider only one relation at maximum for each object!
            #  compute accuracy
            acc_verb, balacc_verb = accuracy_score(y_true=list_gt_verb.cpu().numpy(),
                                                   y_pred=list_pred_verb.cpu().numpy()), balanced_accuracy_score(
                y_true=list_gt_verb.cpu().numpy(), y_pred=list_pred_verb.cpu().numpy())
            acc_rel, balacc_rel = accuracy_score(y_true=list_gt_rel.cpu().numpy(),
                                                 y_pred=list_pred_rel.cpu().numpy()), balanced_accuracy_score(
                y_true=list_gt_rel.cpu().numpy(), y_pred=list_pred_rel.cpu().numpy())
            print(
                f"Validation epoch {epoch}, acc_verb: {acc_verb:.4f}, balAcc_verb: {balacc_verb:.4f}, acc_rel: {acc_rel:.4f}, balAcc_rel: {balacc_rel:.4f}")

            # top-k acc
            ks = [1, 2, 5, 10]
            verb_acc = topk_accuracy(output=list_logits_verb, target=list_gt_verb, topk=ks)
            rel_acc = topk_accuracy(output=list_logits_rel, target=list_gt_rel, topk=ks)
            print(f"\nVerb accuracy:")
            for i in range(len(ks)):
                print(f"top-{ks[i]} accuracy: {verb_acc[i]}")

            print(f"\nRel accuracy:")
            for i in range(len(ks)):
                print(f"top-{ks[i]} accuracy: {rel_acc[i]}")

            if opt.wandb:
                wandb.log({"val/verb_acc": acc_verb, "val/verb_balAcc": balacc_verb, "val/rel_acc": acc_rel,
                           "val/rel_balAcc": balacc_rel, "val/epoch": epoch})

            ##############
            # CHECKPOINT #
            ##############
            save_dir = f"./experiments/{opt.exp_name}/checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, path=osp.join(save_dir, "last.ckpt"))

    if opt.wandb:
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

    model = EASG_VAE(obj_dim,
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
    if args.eval:
        assert args.resume is not None, "eval mode but checkpoint has not been specified"
        model_weights = torch.load(args.resume)['model_state_dict']
        print("Load model weights:\n", model.load_state_dict(model_weights))
        eval(dataloader=val_loader, model=model, device=device, opt=args)
        sys.exit(0)

    if args.scheduler_type == 'cosine_annealing':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_annealing_param)
    elif args.scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        raise Exception('Wrong scheduler type')

    # TRAIN THE MODEL
    train(train_loader=train_loader, val_loader=val_loader, model=model, optimizer=optimizer, scheduler=scheduler,
          device=device, opt=args)


if __name__ == "__main__":
    main()
