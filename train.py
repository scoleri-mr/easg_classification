from run_easg import EASGData
from dataset import myEASGDataset
from pathlib import Path
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from utils import set_wandb_config
from models import EASGClassifier
import torch
from torch import cuda
from torch.optim import Adam
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ann_path', type=str, default='annts_in_new_format/', help='path to annotations')
    parser.add_argument('--data_path', type=str, default='data', help='path to ROI and clip features')
    parser.add_argument('--num_epochs', type=int, default=100, help='total number of epochs')
    parser.add_argument('--hidden_proj_dim', type=int, default=1024, help='hidden dimension for linear projection')
    parser.add_argument('--proj_dim', type=int, default=512, help='final dimension of verb and objects after linear projection')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension for the gnn')
    parser.add_argument('--output_dim', type=int, default=512, help='output dimension of the gnn')
    parser.add_argument('--lr_start', type=float, default=0.01, help='starting learning rate')
    parser.add_argument('--lr_gamma', type=int, default=0.5, help='gamma parameter for lr scheduler')
    parser.add_argument('--lr_step_size', type=int, default=20, help='step size for scheduler')
    parser.add_argument('--edge_criterion', type=str, default='mean', help='define the criterion for edge creation: elementwise mean/max between two adjacent nodes')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='dropout probability for gnn layers')
    args = parser.parse_args()
    return args

def train(train_loader, model, optimizer, scheduler, 
          criterion_edges, criterion_verb, criterion_objs,
          config, 
          num_epochs, device,
          proj_dim, hidden_dim, output_dim, 
          wandb_log = True):
    
    model = model.to(device)
    if wandb_log: wandb.init(project = 'easg_classification', config = config)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        count = 0
        for batch in train_loader:
            count += 1
            batch = batch.to(device)        
            optimizer.zero_grad()
            out_edges, out_verb, out_objs = model(batch.x, batch.edge_index)  
            l1 = criterion_edges(out_edges, batch.y[0])
            l2 = criterion_verb(out_verb.unsqueeze(0), batch.y[1])
            l3 = criterion_objs(out_objs, batch.y[2])
            loss = l1 + l2 + l3
            loss.backward()
            optimizer.step()
            if wandb_log: wandb.log({"loss": loss})      
            total_loss += loss.item()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if wandb_log: wandb.log({"current_lr": current_lr})
        
        # average loss for the epoch
        average_loss = total_loss / count   # correct with batch_size = 1
        if num_epochs >= 20:
            if epoch%10 == 0:
                print(f'Epoch {epoch+1}, Loss: {average_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, Loss: {average_loss:.4f}')
        
    torch.save(model.state_dict(), f'trained_models/edge_classifier{num_epochs}-_mean_pd={proj_dim}_hd={hidden_dim}_outd={output_dim}.pth')
    if wandb_log: wandb.finish()

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

    train_original = EASGData(path_annts, path_data, 'train', verbs, objs, rels)
    train_dataset = myEASGDataset(train_original)
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # DEFINE THE MODEL  AND IT'PARAMETERS
    device = 'cuda' if cuda.is_available() else print('CUDA NOT AVAILABLE')
    obj_dim = 1024                  # original object dimension
    verb_dim = 2304                 # original verb dimension

    if args.edge_criterion == 'mean':
        edge_criterion = 'mean'
    elif args.edge_criterion == 'max':
        edge_criterion = 'max'
    else:
        raise Exception('Wrong edge criterion')
    
    edge_classifier = EASGClassifier(obj_dim, verb_dim, 
                                     num_rels, num_verbs, num_objs, 
                                     args.hidden_proj_dim, args.proj_dim, args.hidden_dim, args.output_dim, 
                                    device, args.dropout_prob, edge_criterion)
    optimizer = Adam(edge_classifier.parameters(), lr=args.lr_start)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    criterion_edges = nn.BCEWithLogitsLoss()
    criterion_verb = nn.CrossEntropyLoss()
    criterion_objs = nn.CrossEntropyLoss()
    config = set_wandb_config(args.num_epochs, args.hidden_proj_dim, args.proj_dim, 
                              args.hidden_dim, args.output_dim, batch_size,
                              args.lr_start, args.lr_step_size, args.lr_gamma)

    # TRAIN THE MODEL
    train(train_loader, edge_classifier, optimizer, scheduler, 
          criterion_edges, criterion_verb, criterion_objs, 
          config, args.num_epochs, device, 
          args.proj_dim, args.hidden_dim, args.output_dim)

if __name__ == "__main__":
    main()