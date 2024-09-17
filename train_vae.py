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
from autoencoder import EASGvae
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
from torch.utils.data import Subset
import pickle
from utils import load_model

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--ann_path', type=str, default='./annts_in_new_format/', help='path to annotations')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to ROI and clip features')
    parser.add_argument('--num_epochs', type=int, default=1000, help='total number of epochs')
    parser.add_argument('--beta', type=float, default=0.0005, help='beta weighting kld of vae. beta=1 triggers weighted beta')
    parser.add_argument('--kld_type', type=str, default='original', help='type of kld. Choose between original, mean, commonScenes')
    parser.add_argument('--hidden_proj_dim', type=int, default=1024, help='hidden dimension for linear projection')
    parser.add_argument('--proj_dim', type=int, default=512, help='final dimension of verb and objects after linear projection')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension for the gnn')
    parser.add_argument('--output_dim', type=int, default=256, help='output dimension of the gnn')
    parser.add_argument('--scheduler_type', type=str, default='cosine_annealing', help='choose between step, cosine_annealing and fixed')
    parser.add_argument('--lr_start', type=float, default=0.0001, help='starting learning rate')
    parser.add_argument('--lr_gamma', type=int, default=0.5, help='gamma parameter for lr scheduler')
    parser.add_argument('--lr_step_size', type=int, default=20, help='step size for scheduler')
    parser.add_argument('--eps', type=float, default= 1., help='set the epsilon for the vae reparametrize')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='dropout probability for gnn layers')
    parser.add_argument('--wandb', action='store_true', help="If specified enables wandb logging")
    parser.add_argument('--graph_type', type=str, default='gcn', help='choose between graph layers: gcn, sage, gat, gin')
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--check_overfitting', action='store_true', help="If specified takes a random subset of the training set to check overfitting capabilities of the model")
    parser.add_argument('--focal_loss', action='store_true', help="If specified use focal loss to balance verb classes")
    parser.add_argument('--exclude_verbs', action='store_true', help="If specified exclude verbs from training, use to focus on relationships")
    parser.add_argument('--wandb_proj', type=str, default='vae_balanced_losses')
    parser.add_argument('--separate', action='store_true', help='If specified separates the heads of verbs and relationships removing common mpl in the decoder')
    parser.add_argument('--from_ae', action='store_true', help='if specified start training from ae in base_path')
    parser.add_argument('--base_path', type=str, help='path of the starting pretrained model')
    parser.add_argument('--balance_losses', action='store_true', help="If specified balances losses dynamically")
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
    plt.savefig(f'plots/losses_vae_kld={opt.kld_type}_b={opt.beta}_ld={opt.output_dim}_lr={opt.lr_start}.png')
    plt.show()

def weight_beta(num_epochs, beta):
    if beta==1:
        x = np.linspace(-15, 6, num_epochs)
        weights = (1 / (1 + np.exp(-x)))*0.05
    else:
        weights = np.ones(num_epochs)
    return weights

def train(train_loader, val_loader, model, optimizer, scheduler, device, opt):
    if opt.exp_name is None:
        opt.exp_name = f"VAE{opt.num_epochs}_balancedL_{opt.graph_type}_sep={opt.separate}_fromae={opt.from_ae}_od={opt.output_dim}_kld={opt.kld_type}_b={opt.beta}_drop={opt.dropout_prob}_lr={opt.lr_start}_fl={opt.focal_loss}_ex={opt.exclude_verbs}_eps={opt.eps}_{str(int(time.time()))}"
    print(f"Training - exp name: {opt.exp_name}")        
        
    model = model.to(device)
    logger = logging.getLogger()
    w = weight_beta(opt.num_epochs, opt.beta)

    if opt.wandb:
        wandb.init(project=f'{opt.wandb_proj}', config=opt, name=opt.exp_name)
        wandb.watch(model, log="all")        

    history_verb = []
    history_rels = []
    history_kld = []

    log_filename = f'log_{opt.exp_name}'
    log_file_path = os.path.join('./experiments/vae_manualWeights', log_filename)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[logging.StreamHandler(), logging.FileHandler(filename=log_file_path, mode='w')],
                        )
    count=0
    for epoch in range(opt.num_epochs):
        model.train()
        for bidx, _data in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
            batch, verb_gt, rel_gt = _data
            batch = batch.to(device)
            verb_gt = verb_gt.view(-1).to(device)  # [bs, ]
            rel_gt = rel_gt.to(device)  # [bs, num_objs, num_rels+1]
            optimizer.zero_grad()
            verb_logits, relationship_logits, mu, logvar = model(batch)
            if opt.balance_losses:
                # Loss scaling based on uncertainty
                loss_verb_weighted = loss_verb / (2 * model.sigma_verb**2) + torch.log(model.sigma_verb)
                loss_rel_weighted = loss_rel / (2 * model.sigma_rel**2) + torch.log(model.sigma_rel)
                loss_kld_weighted = opt.beta * kld * w[epoch] / (2 * model.sigma_kld**2) + torch.log(model.sigma_kld)

                # Final loss
                total_loss = loss_verb_weighted + loss_rel_weighted + loss_kld_weighted
            else: 
                loss_verb, loss_rel, kld = model.loss_functions(verb_gt, rel_gt, verb_logits, relationship_logits, mu, logvar)
                total_loss = loss_verb + loss_rel + opt.beta*kld*w[epoch]
                
            history_verb.append(loss_verb.item())
            history_rels.append(loss_rel.item())
            history_kld.append(kld.item())
            total_loss.backward()
            optimizer.step()
            if bidx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                # print(f"Train epoch {epoch}, it: {bidx}, loss: {loss.item():.4f}, loss_verb: {loss_verb.item():.4f}, loss_rel: {loss_rel.item():.4f}, kld: {kld.item()}")
                if opt.wandb: wandb.log({"loss": total_loss, "loss_verb": loss_verb, "loss_rel": loss_rel, "kdl":kld, "current_lr": current_lr})                 
        scheduler.step()
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            # EVALUATION
            if opt.check_overfitting:
                val_loader = train_loader

            acc_verb, balacc_verb, acc_rel, balacc_rel, topk_acc_verb, topk_acc_rels = evaluation(model, val_loader, device, k_list_verbs=[1,2,5,10,20])
            acc_verb_t, balacc_verb_t, acc_rel_t, balacc_rel_t, topk_acc_verb_t, topk_acc_rels_t = evaluation(model, train_loader, device, k_list_verbs = [1,2,5,10,20])
            local_logging(logger, acc_verb, balacc_verb, acc_verb_t, balacc_verb_t, topk_acc_verb, topk_acc_rels, topk_acc_verb_t, topk_acc_rels_t, epoch+1, [1,2,5,10])

            if opt.wandb: 
                wandb.log({"epoch": epoch, "acc_verb_val": acc_verb, "balacc_verb_val": balacc_verb, "topk_acc_verb_val":topk_acc_verb, "topk_acc_rels_val":topk_acc_rels,
                        "acc_verb_train": acc_verb_t, "balacc_verb_train": balacc_verb_t, 
                        'acc_rels': acc_rel, 'balacc_rels': balacc_rel, 'acc_rels_train': acc_rel_t, 'balacc_rels_train': balacc_rel_t})
            
            # recap excel file
            if epoch+1==opt.num_epochs:
                log_run_to_excel(opt, acc_verb, balacc_verb, topk_acc_verb, topk_acc_rels)

            # CHECKPOINT
            save_dir = f"./experiments/vae_balanced_losses/{opt.exp_name}/checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, path=osp.join(save_dir, "last.ckpt"))

    plot_losses(history_verb, history_rels, history_kld, opt)
    if opt.wandb: 
        wandb.log({"epoch": epoch, "acc_verb_val": acc_verb, "balacc_verb_val": balacc_verb, "topk_acc_verb_val":topk_acc_verb, "topk_acc_rels_val":topk_acc_rels,
                   "acc_verb_train": acc_verb_t, "balacc_verb_train": balacc_verb_t})

    if opt.wandb:
        wandb.finish()

def log_run_to_excel(opt, acc_verb, balacc_verb, topk_acc_verb, topk_acc_rels, file_path='vae_runs.xlsx'):
    import pandas as pd
    from openpyxl import load_workbook
    topk_acc_verb[1].item()
    run_data = {
        'learning_rate': str(opt.lr_start),
        'batch_size': opt.batch_size,
        'dropout': opt.dropout_prob,
        'kld_type': opt.kld_type,
        'beta': str(opt.beta),
        'latent_dim': opt.output_dim,
        'epochs': opt.num_epochs,
        'focal_loss': opt.focal_loss,
        'acc_verb': str("{:.4f}".format(acc_verb)),
        'balacc_verb': str("{:.4f}".format(balacc_verb)), 
        'top1_vacc': str("{:.4f}".format(topk_acc_verb[1].item())),
        'top2_vacc': str("{:.4f}".format(topk_acc_verb[2].item())),
        'top5_vacc': str("{:.4f}".format(topk_acc_verb[5].item())),
        'top10_vacc': str("{:.4f}".format(topk_acc_verb[10].item())),
        'top1_racc': str("{:.4f}".format(topk_acc_rels[1].item())),
        'top2_racc': str("{:.4f}".format(topk_acc_rels[2].item())),
        'top5_racc': str("{:.4f}".format(topk_acc_rels[5].item())),
        'top10_racc': str("{:.4f}".format(topk_acc_rels[10].item())),
    }
    df = pd.DataFrame([run_data])

    # Check if the Excel file already exists
    try:
        # Load existing Excel file
        existing_df = pd.read_excel(file_path)
        
        # Append new data to existing DataFrame
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        print('Creating excel file')
        pass  # Excel file does not exist, so no need to append
    
    # Save DataFrame to Excel file
    df.to_excel(file_path, index=False)

def local_logging(logger, acc_verb, balacc_verb, acc_verb_train, balacc_verb_train, topk_acc_verb, topk_acc_rels, topk_acc_verb_train, topk_acc_rels_train, epoch, list_k):
    logger.info(f"VALIDATION EPOCH {epoch}:")
    logger.info(f"acc_verb: {acc_verb}, balacc_verb: {balacc_verb}")
    logger.info(f"acc_verb_train: {acc_verb_train}, balacc_verb_train: {balacc_verb_train}")
    logger.info(f"topk verb accuracy {list_k}: {topk_acc_verb[list_k[0]].item():.4f}, {topk_acc_verb[list_k[1]].item():.4f}, {topk_acc_verb[list_k[2]].item():.4f}, {topk_acc_verb[list_k[3]].item():.4f}")
    logger.info(f"topk rels accuracy {list_k}: {topk_acc_rels[list_k[0]].item():.4f}, {topk_acc_rels[list_k[1]].item():.4f}, {topk_acc_rels[list_k[2]].item():.4f}, {topk_acc_rels[list_k[3]].item():.4f}")
    logger.info(f"topk verb accuracy train {list_k}: {topk_acc_verb_train[list_k[0]].item():.4f}, {topk_acc_verb_train[list_k[1]].item():.4f}, {topk_acc_verb_train[list_k[2]].item():.4f}, {topk_acc_verb_train[list_k[3]].item():.4f}")
    logger.info(f"topk rels accuracy train{list_k}: {topk_acc_rels_train[list_k[0]].item():.4f}, {topk_acc_rels_train[list_k[1]].item():.4f}, {topk_acc_rels_train[list_k[2]].item():.4f}, {topk_acc_rels_train[list_k[3]].item():.4f}")
    logger.info("\n")

def evaluation(model, val_loader, device, k_list_verbs = [1,2,5,10], k_list_rels = [1,2,5,10]):
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

    # VERB ACCURACIES
    # total verb accuracy and balanced verb accuracy 
    list_logits_verb = torch.cat(list_logits_verb, dim=0)
    list_pred_verb = torch.argmax(list_logits_verb, -1)
    list_gt_verb = torch.cat(list_gt_verb, dim=0)
    acc_verb = accuracy_score(y_true=list_gt_verb.cpu().numpy(), y_pred=list_pred_verb.cpu().numpy())
    balacc_verb = balanced_accuracy_score(y_true=list_gt_verb.cpu().numpy(), y_pred=list_pred_verb.cpu().numpy())
    # topk verb accuracy
    topk_acc_verb = topk_accuracy(output=list_logits_verb, target=list_gt_verb, topk=k_list_verbs)
    topk_acc_verb = dict(zip(k_list_verbs, topk_acc_verb))

    # RELATIONSHIP ACCURACIES:
    # only the topk accuracy makes sense here since we can have more than one relationship
    list_logits_rels = torch.cat(list_logits_rel, dim=0).view(-1, 14)  # [total_instances, 14]
    list_gt_rels = torch.cat(list_gt_rels, dim=0)  # [total_instances]
    relationship_accuracies = {k: [] for k in k_list_rels}
    topk_acc_rels = {k: [] for k in k_list_rels}
    for i in range(list_logits_rels.shape[0]):
        gt_relations = list_gt_rels[i]
        if gt_relations != 13:  # Exclude the 13th relationship
            for k in k_list_rels:
                topk_predictions = torch.topk(list_logits_rels[i, :-1], k, dim=0).indices
                correct = (topk_predictions == gt_relations).any().item()
                relationship_accuracies[k].append(correct)

    for k in k_list_rels:
        topk_acc_rels[k] = np.mean(relationship_accuracies[k])

    # get a balanced accuracy for relationships as well. WARNING: this balanced accuracy does not take into
    # account the possibility to have multiple relationship, it's just to see how the focal loss changes the results
    list_pred_rels = torch.argmax(list_logits_rels,-1)
    acc_rel= accuracy_score(y_true=list_gt_rels.cpu().numpy(), y_pred=list_pred_rels.cpu().numpy())
    balacc_rel = balanced_accuracy_score(y_true=list_gt_rels.cpu().numpy(), y_pred=list_pred_rels.cpu().numpy())

    return acc_verb, balacc_verb, acc_rel, balacc_rel, topk_acc_verb, topk_acc_rels

def main():
    torch.manual_seed(42)

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
    if args.check_overfitting:
        train_dataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:300])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) #, drop_last=False)  #shuffle=True)
    
    if args.check_overfitting:
        validation_dataset = train_dataset
    else:
        validation_dataset = EASGDatasetAE(path_annts, path_data, 'val', verbs, objs, rels)      
    val_loader = DataLoader(validation_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)

    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(validation_dataset)} samples")
    
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
                    args.graph_type,
                    args.focal_loss,
                    args.eps,
                    args.separate,
                    args.balance_losses
                    )
    optimizer = Adam(model.parameters(), lr=args.lr_start)

    if args.from_ae:
        assert len(args.base_path) > 0, "if training from base model, we need the model path"
        print("Start training the vae from specified ae model")
        model.load_state_dict(torch.load(args.base_path)['model_state_dict'], strict=False)

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