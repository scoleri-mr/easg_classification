import os
import os.path as osp
import sys
import numpy as np
from dataset_ae import EASGDatasetAE
from pathlib import Path
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from autoencoder_mine import EASGAutoEncoder, EASGvae
import torch
from torch import cuda
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import logging

""""
example launcher: python train_ae.py --wandb --exp_name AE_verb_rel_withVal_epochs200 --num_epochs 200
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--ann_path', type=str, default='./annts_in_new_format/', help='path to annotations')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to ROI and clip features')
    parser.add_argument('--num_epochs', type=int, default=100, help='total number of epochs')
    parser.add_argument('--beta', type=float, default=0.05, help='beta weighting kld of vae. beta=1 triggers weighted beta')
    parser.add_argument('--kld_type', type=str, default='original', help='type of kld. Choose between original, mean, commonScenes')
    parser.add_argument('--hidden_proj_dim', type=int, default=1024, help='hidden dimension for linear projection')
    parser.add_argument('--proj_dim', type=int, default=512, help='final dimension of verb and objects after linear projection')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension for the gnn')
    parser.add_argument('--output_dim', type=int, default=512, help='output dimension of the gnn')
    parser.add_argument('--scheduler_type', type=str, default='cosine_annealing', help='choose between step, cosine_annealing and fixed')
    parser.add_argument('--lr_start', type=float, default=0.0001, help='starting learning rate')
    parser.add_argument('--lr_gamma', type=int, default=0.5, help='gamma parameter for lr scheduler')
    parser.add_argument('--lr_step_size', type=int, default=20, help='step size for scheduler')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='dropout probability for gnn layers')
    parser.add_argument('--wandb', action='store_true', help="If specified enables wandb logging")
    parser.add_argument('--graph_type', type=str, default='gcn', help='choose between graph layers: gcn, sage, gat, gin')
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    return args

def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

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

def plot_losses(loss_verb, loss_rels, kld, opt):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(loss_verb, label='Loss verb', color='blue')
    axs[0].set_title('Loss verb')
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(loss_rels, label='Relationship loss', color='green')
    axs[1].set_title('Relationship loss')
    axs[1].set_xlabel('steps')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    axs[2].plot(kld, label='kld', color='red')
    axs[2].set_title('kld')
    axs[2].set_xlabel('steps')
    axs[2].set_ylabel('Loss')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f'plots/losses_vae_kld={opt.kld_type}_b={opt.beta}_ld={opt.output_dim}.png')
    plt.show()

def weight_beta(num_epochs, beta):
    if beta==1:
        x = np.linspace(-6, 6, num_epochs)
        weights = (1 / (1 + np.exp(-x)))*0.05
    else:
        weights = np.ones(num_epochs)
    return weights

def train(train_loader, val_loader, model, optimizer, scheduler, device, opt):
    if opt.exp_name is None:
        opt.exp_name = f"VAE_od={opt.output_dim}_kld={opt.kld_type}_b={opt.beta}_{str(int(time.time()))}"
    print(f"Training - exp name: {opt.exp_name}")        
        
    model = model.to(device)
    logger = logging.getLogger()
    w = weight_beta(opt.num_epochs, opt.beta)

    if opt.wandb:
        wandb.init(project=f'vae_easg', config=opt, name=opt.exp_name)
        wandb.watch(model, log="all")        

    history_verb = []
    history_rels = []
    history_kld = []

    log_filename = f'log_vae_{opt.kld_type}_b={opt.beta}_ld={opt.output_dim}'
    log_file_path = os.path.join('./experiments', log_filename)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[logging.StreamHandler(), logging.FileHandler(filename=log_file_path, mode='w')],
                        )

    for epoch in range(opt.num_epochs):
        model.train()
        for bidx, _data in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
            batch, verb_gt, rel_gt = _data
            batch = batch.to(device)
            verb_gt = verb_gt.view(-1).to(device)  # [bs, ]
            rel_gt = rel_gt.to(device)  # [bs, num_objs, num_rels+1]
            optimizer.zero_grad()
            verb_logits, relationship_logits, mu, logvar = model(batch)
            loss_verb, loss_rel, kld = model.loss_functions(verb_gt, rel_gt, verb_logits, relationship_logits, mu, logvar)
            loss = loss_verb + loss_rel + opt.beta*kld*w[epoch]
            history_verb.append(loss_verb.item())
            history_rels.append(loss_rel.item())
            history_kld.append(kld.item())
            loss.backward()
            optimizer.step()
            if bidx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                # print(f"Train epoch {epoch}, it: {bidx}, loss: {loss.item():.4f}, loss_verb: {loss_verb.item():.4f}, loss_rel: {loss_rel.item():.4f}, kld: {kld.item()}")
                if opt.wandb: wandb.log({"loss": loss, "loss_verb": loss_verb, "loss_rel": loss_rel, "kdl":kld, "current_lr": current_lr})                 
        scheduler.step()
        
        if epoch % 10 == 0:
            # EVALUATION
            # acc_verb, balacc_verb, acc_rel, balacc_rel, verb_acc_topk, rel_acc_topk = evaluation(model, val_loader, device, epoch)
            # local_logging(logger, epoch, acc_verb, balacc_verb, acc_rel, balacc_rel, verb_acc_topk, rel_acc_topk)

            # if opt.wandb: 
            #     wandb.log({"val/verb_acc": acc_verb, "val/verb_balAcc": balacc_verb, "val/rel_acc": acc_rel, "val/rel_balAcc": balacc_rel, "val/epoch": epoch})
            
            # CHECKPOINT
            # save_dir = f"./experiments/{opt.exp_name}/checkpoints"
            # os.makedirs(save_dir, exist_ok=True)
            # save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, path=osp.join(save_dir, "last.ckpt"))

            # final_verb_accuracies, final_relationship_accuracies = evaluate_model(model, val_loader, epoch, device)
            # local_logging2(logger, epoch, final_verb_accuracies, final_relationship_accuracies)

            evaluation_mine(model, val_loader, epoch, device)

    # acc_verb, balacc_verb, acc_rel, balacc_rel, verb_acc_topk, rel_acc_topk = evaluation(model, val_loader, device, epoch)
    # local_logging(logger, epoch, acc_verb, balacc_verb, acc_rel, balacc_rel, verb_acc_topk, rel_acc_topk)
    plot_losses(history_verb, history_rels, history_kld, opt)

    if opt.wandb:
        wandb.finish()

def local_logging(logger, epoch, acc_verb, balacc_verb, acc_rel, balacc_rel, verb_acc_topk, rel_acc_topk):
    logger.info(f'EPOCH {epoch}')
    logger.info(f'VALIDATION: verb_acc={acc_verb}, verb_balAcc={balacc_verb}, rel_acc:{acc_rel}, rel_balAcc:{balacc_verb} ')
    logger.info(f'topk verb accuracy [1,2,5,10]: {verb_acc_topk[0].item():.4f}, {verb_acc_topk[1].item():.4f}, {verb_acc_topk[2].item():.4f}, {verb_acc_topk[3].item():.4f}')
    logger.info(f'topk rel accuracy [1,2,5,10]: {rel_acc_topk[0].item():.4f}, {rel_acc_topk[1].item():.4f}, {rel_acc_topk[2].item():.4f}, {rel_acc_topk[3].item():.4f}')
    logger.info('\n')

def local_logging2(logger, epoch, verb_topk, rels_topk):
    logger.info(f"VALIDATION EPOCH {epoch}:")
    logger.info(f"topk verb accuracy [1,2,5,10]: {verb_topk['top1']:.4f}, {verb_topk['top2']:.4f}, {verb_topk['top5']:.4f}, {verb_topk['top10']:.4f}")
    logger.info(f"topk rels accuracy [1,2,5,10]: {rels_topk['top1']:.4f}, {rels_topk['top2']:.4f}, {rels_topk['top5']:.4f}, {rels_topk['top10']:.4f}")
    logger.info("\n")

def compute_topk_balanced_accuracy(verbs_logits, rels_pred, verb_gt, rels_gt, k_list=[1, 2, 5, 10]):
    bs = verbs_logits.size(0)
    num_objects = 391

    verbs_logits = verbs_logits.cpu().detach()
    rels_pred = rels_pred.cpu().detach()
    
    # Exclude the 13th relationship
    rels_pred = rels_pred[:, :, :13]
    rels_gt = rels_gt[:, :, :13]

    # Verb top-k accuracy
    topk_accuracies = {}
    for k in k_list:
        topk_predictions = torch.topk(verbs_logits, k, dim=1).indices
        topk_correct = topk_predictions.eq(verb_gt.view(-1, 1).expand_as(topk_predictions))
        topk_correct_any = topk_correct.any(dim=1)
        topk_balanced_acc = balanced_accuracy_score(torch.ones_like(topk_correct_any).cpu(), topk_correct_any.cpu())
        topk_accuracies[f'top{k}'] = topk_balanced_acc

    # Relationship top-k accuracy
    relationship_topk_accuracies = {}
    for k in k_list:
        relationship_accuracies = []
        for b in range(bs):
            for i in range(num_objects):
                if torch.sum(rels_gt[b, i, :]) > 0:  # Check if object is present
                    topk_predictions = torch.topk(rels_pred[b, i, :], k, dim=0).indices
                    gt_relations = torch.nonzero(rels_gt[b, i, :], as_tuple=False).squeeze()
                    correct = torch.any(torch.eq(topk_predictions.view(-1, 1), gt_relations.view(1, -1)), dim=1).any().item()
                    relationship_accuracies.append(correct)
        relationship_accuracies = torch.tensor(relationship_accuracies)
        balanced_relationship_acc = balanced_accuracy_score(torch.ones_like(relationship_accuracies).cpu(), relationship_accuracies.cpu())
        relationship_topk_accuracies[f'top{k}'] = balanced_relationship_acc

    return topk_accuracies, relationship_topk_accuracies

def evaluate_model(model, val_loader, epoch, device, k_list=[1, 2, 5, 10]):
    total_verb_accuracies = {f'top{k}': [] for k in k_list}
    total_relationship_accuracies = {f'top{k}': [] for k in k_list}
    model.eval()
    with torch.no_grad():
        for batch, verb_gt, rel_gt in val_loader:
            batch = batch.to(device)
            verb_logits, rels_pred, _, _ = model(batch)
            topk_accuracies, relationship_topk_accuracies = compute_topk_balanced_accuracy(
                verb_logits, rels_pred, verb_gt, rel_gt, k_list
            )
            
            for k in k_list:
                total_verb_accuracies[f'top{k}'].append(topk_accuracies[f'top{k}'])
                total_relationship_accuracies[f'top{k}'].append(relationship_topk_accuracies[f'top{k}'])

    # Compute the mean accuracy for each k
    final_verb_accuracies = {k: torch.tensor(total_verb_accuracies[k]).mean().item() for k in total_verb_accuracies}
    final_relationship_accuracies = {k: torch.tensor(total_relationship_accuracies[k]).mean().item() for k in total_relationship_accuracies}

    print(f'Epoch {epoch} accuracy:')
    for i in range(len(k_list)):
        print(f"top-{k_list[i]} accuracy: {final_verb_accuracies[f'top{k_list[i]}']}")
        
    print(f"\nRel accuracy:")
    for i in range(len(k_list)):
        print(f"top-{k_list[i]} accuracy: {final_relationship_accuracies[f'top{k_list[i]}']}")

    return final_verb_accuracies, final_relationship_accuracies

def evaluation_mine(model, val_loader, epoch, device, k_list = [1,2,5,10]):
    model = model.eval()
    list_logits_verb, list_gt_verb = [], []
    list_logits_rel, list_gt_rels = [], []

    for _data in val_loader:
        batch, verb_gt, rel_gt = _data
        batch = batch.to(device)
        verb_gt = verb_gt.view(-1).to(device)  # [bs, ]
        rel_gt = rel_gt.to(device)  # [bs, num_objs, num_rels+1]
        out_verb, out_rel, _, _ = model(batch)
        out_rel = out_rel.contiguous().view(-1, rel_gt.size(1), 14)
        # store val batch results for computing global accuracy and balanced accuracy
        list_logits_verb.append(out_verb.cpu().detach())
        list_logits_rel.append(out_rel.cpu().detach())
        list_gt_verb.append(verb_gt.cpu().detach())
        list_gt_rels.append(rel_gt.view(-1,14).argmax(-1).view(-1).cpu().detach())

    print(f'VALIDATION EPOCH {epoch}:')
    # VERB ACCURACIES
    # total verb accuracy and balanced verb accuracy 
    list_logits_verb = torch.cat(list_logits_verb, dim=0)
    list_pred_verb = torch.argmax(list_logits_verb, -1)
    list_gt_verb = torch.cat(list_gt_verb, dim=0)
    acc_verb = accuracy_score(y_true=list_gt_verb.cpu().numpy(), y_pred=list_pred_verb.cpu().numpy())
    balacc_verb = balanced_accuracy_score(y_true=list_gt_verb.cpu().numpy(), y_pred=list_pred_verb.cpu().numpy())
    print(f"acc_verb: {acc_verb:.4f}, balacc_verb: {balacc_verb:.4f}")
    # topk verb accuracy
    verb_acc_topk = topk_accuracy(output=list_logits_verb, target=list_gt_verb, topk=k_list)
    for i in range(len(k_list)):
        print(f"top-{k_list[i]} verb accuracy: {verb_acc_topk[i]}")

    # RELATIONSHIP ACCURACIES:
    # only the topk accuracy makes sense here since we can have more than one relationship

    # Exclude the 13th relationship
    list_logits_rels = torch.cat(list_logits_rel, dim=0)  # [total_instances, 14]
    list_gt_rels = torch.cat(list_gt_rels, dim=0)  # [total_instances]

    relationship_accuracies = []
    for i in range(list_logits_rels.shape[0]):
        if list_gt_rels[i] != 13:  # If the object is present
            topk_predictions = torch.topk(list_logits_rels[i, :-1], max(k_list), dim=0).indices
            gt_relations = list_gt_rels[i]
            correct = (topk_predictions == gt_relations).any().item()
            relationship_accuracies.append(correct)

    for k in k_list:
        topk_acc = np.mean([(torch.topk(list_logits_rels[i, :-1], k, dim=0).indices == list_gt_rels[i]).any().item() for i in range(list_logits_rels.shape[0]) if list_gt_rels[i] != 13])
        print(f"top-{k} relationship accuracy: {topk_acc:.4f}")


def evaluation_alli(model, val_loader, device, epoch):
    model = model.eval()
    list_logits_verb, list_gt_verb = [], []
    list_logits_rel, list_gt_rel = [], []
    for bidx, _data in tqdm(enumerate(val_loader, 0), unit="batch", total=len(val_loader), desc="Validation"):
        batch, verb_gt, rel_gt = _data
        batch = batch.to(device)
        verb_gt = verb_gt.view(-1).to(device)  # [bs, ]
        rel_gt = rel_gt.to(device)  # [bs, num_objs, num_rels+1]
        out_verb, out_rel, _, _ = model(batch)
        out_rel = out_rel.contiguous().view(-1, 14)
        # store val batch results for computing global accuracy and balanced accuracy
        list_logits_verb.append(out_verb.cpu().detach())
        list_logits_rel.append(out_rel.cpu().detach())
        list_gt_verb.append(verb_gt.cpu().detach())
        list_gt_rel.append(rel_gt.view(-1,14).argmax(-1).view(-1).cpu().detach())
    
    list_logits_verb = torch.cat(list_logits_verb, dim=0)
    list_pred_verb = torch.argmax(list_logits_verb, -1)
    list_logits_rel = torch.cat(list_logits_rel, dim=0)
    list_pred_rel = torch.argmax(list_logits_rel, -1)  # TODO: this does not address multiple verb-obj relationships!
    list_gt_verb = torch.cat(list_gt_verb, dim=0)
    list_gt_rel = torch.cat(list_gt_rel, dim=0)
    
    # TODO: during training the accuracy of realtions is not 100% correct - we consider only one relation at maximum for each object!
    # compute accuracy
    acc_verb, balacc_verb = accuracy_score(y_true=list_gt_verb.cpu().numpy(), y_pred=list_pred_verb.cpu().numpy()), balanced_accuracy_score(y_true=list_gt_verb.cpu().numpy(), y_pred=list_pred_verb.cpu().numpy())
    acc_rel, balacc_rel = accuracy_score(y_true=list_gt_rel.cpu().numpy(), y_pred=list_pred_rel.cpu().numpy()), balanced_accuracy_score(y_true=list_gt_rel.cpu().numpy(), y_pred=list_pred_rel.cpu().numpy())
    print(f"Validation epoch {epoch}, acc_verb: {acc_verb:.4f}, balAcc_verb: {balacc_verb:.4f}, acc_rel: {acc_rel:.4f}, balAcc_rel: {balacc_rel:.4f}")
    
    # top-k acc
    ks = [1,2,5,10]
    verb_acc_topk = topk_accuracy(output=list_logits_verb, target=list_gt_verb, topk=ks)
    rel_acc_topk = topk_accuracy(output=list_logits_rel, target=list_gt_rel, topk=ks)
    print(f"\nVerb accuracy:")
    for i in range(len(ks)):
        print(f"top-{ks[i]} accuracy: {verb_acc_topk[i]}")
        
    print(f"\nRel accuracy:")
    for i in range(len(ks)):
        print(f"top-{ks[i]} accuracy: {rel_acc_topk[i]}")

    return acc_verb, balacc_verb, acc_rel, balacc_rel, verb_acc_topk, rel_acc_topk

def main():
    # get datasets
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) #, drop_last=False)  #shuffle=True)

    validation_dataset = EASGDatasetAE(path_annts, path_data, 'val', verbs, objs, rels)
    val_loader = DataLoader(validation_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)

    # DEFINE THE MODEL  AND IT'PARAMETERS
    device = 'cuda' if cuda.is_available() else print('CUDA NOT AVAILABLE')
    obj_dim = 1024  # original object dimension
    verb_dim = 2304  # original verb dimension

    cosine_annealing_param = args.num_epochs

    model = EASGvae(obj_dim,
                    verb_dim,
                    num_rels + 1,
                    num_verbs, num_objs,
                    args.hidden_proj_dim,
                    args.proj_dim,
                    args.hidden_dim,
                    args.output_dim,
                    args.kld_type,
                    args.dropout_prob,
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
    # TODO: better to have a train_one_epoch() fun.
    train(train_loader=train_loader, val_loader=val_loader, model=model, optimizer=optimizer, scheduler=scheduler, device=device, opt=args)

if __name__ == "__main__":
    main()