from src.scripts.run_easg import EASGData
from src.data.datasets.dataset import myEASGDataset
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
from math import ceil
import logging
import pickle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str, help='provide model path and name')
    parser.add_argument('--ann_path', type=str, default='annts_in_new_format/', help='path to annotations')
    parser.add_argument('--partition', type=str, default='val', help='dataset partitio to evaluate, use train to check overfitting and val to evaluate the model')
    parser.add_argument('--data_path', type=str, default='data', help='path to ROI and clip features')
    parser.add_argument('--hidden_proj_dim', type=int, default=1024, help='hidden dimension for linear projection')
    parser.add_argument('--proj_dim', type=int, default=512, help='final dimension of verb and objects after linear projection')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension for the gnn')
    parser.add_argument('--output_dim', type=int, default=512, help='output dimension of the gnn')
    parser.add_argument('--edge_criterion', type=str, default='conc', help='define the criterion for edge creation: elementwise mean/max between two adjacent nodes')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='dropout probability for gnn layers')
    parser.add_argument('--graph_type', type=str, default='gcn', help='choose between graph layers: gcn, sage, gat')
    args = parser.parse_args()
    return args

from math import ceil
import torch 

def evaluation(dataset_val, model, device, dump_output=False):
    # NB: with significa with constraints (ovvero si vincola il grafo ad avere al massimo
    # una relazione object-verb), no significa No constraint (quindi niente vincolo sul 
    # numero di possibili relazioni.)

    def intersect_2d(out, gt):
        '''search each gt triplet in the out triplets. Returns a tensor the same size as out in which 
        we have true in the position corresponding to the triplet matching the gt'''
        return (out[..., None] == gt.T[None, ...]).all(1)
    
    model.eval()

    num_top_verb = 5
    num_top_rel_with = 1
    num_top_rel_no = 5
    list_k = [10, 20, 50]

    recall_predcls_with = {k: [] for k in list_k}
    recall_predcls_no = {k: [] for k in list_k}
    recall_sgcls_with = {k: [] for k in list_k}
    recall_sgcls_no = {k: [] for k in list_k}
    recall_easgcls_with = {k: [] for k in list_k}
    recall_easgcls_no = {k: [] for k in list_k}
    verbs_predictions = []
    for idx in range(len(dataset_val)):
        graph = dataset_val[idx].to(device)

        with torch.no_grad():
            logits_edges, logits_verb, logits_objs = model(graph.x, graph.edge_index)
            scores_verb = logits_verb[0].detach().cpu().softmax(dim=0).to(device)
            scores_objs = logits_objs.detach().cpu().softmax(dim=1).to(device)
            verbs_predictions.append(scores_verb)
            scores_rels = logits_edges.detach().cpu().sigmoid().to(device)

        verb_idx = dataset_val.get_verb_index(idx).to(device)
        obj_indices = dataset_val.get_object_indices(idx).to(device)
        num_obj = obj_indices.shape[0]
        rels_vecs = dataset_val.get_rels(idx).to(device)
        triplets_gt = dataset_val.get_original_triplets(idx).to(device)
        
        # if len(obj_indices)==1:
        #     triplets_gt[0][1] = triplets_gt[0][1] - 198
        #     if triplets_gt[0][1] < 0: 
        #         raise Exception('problem with object index!')

        # make triplets for verb CONTINUE THIS CODE...
        triplets_verb_with = [] 
        scores_verb_with = []
        triplets_verb_no = []
        scores_verb_no = []
        num_top_obj_with = ceil(max(list_k)/(num_top_verb*num_top_rel_with*num_obj))
        num_top_obj_no = ceil(max(list_k)/(num_top_verb*num_top_rel_no*num_obj))
        for vi in scores_verb.argsort(descending=True)[:num_top_verb]:
            for scores_obj, scores_rel in zip(scores_objs, scores_rels):
                sorted_scores_obj = scores_obj.argsort(descending=True)
                sorted_scores_rel = scores_rel.argsort(descending=True)
                for oi in sorted_scores_obj[:num_top_obj_with]:
                    for ri in sorted_scores_rel[:num_top_rel_with]:
                        triplets_easg_with.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_with.append((scores_verb[vi]+scores_obj[oi]+scores_rel[ri]).item())
                for oi in sorted_scores_obj[:num_top_obj_no]:
                    for ri in sorted_scores_rel[:num_top_rel_no]:
                        triplets_easg_no.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_no.append((scores_verb[vi]+scores_obj[oi]+scores_rel[ri]).item())

        # make triplets for precls
        triplets_pred_with = []
        scores_pred_with = []
        triplets_pred_no = []
        scores_pred_no = []
        for obj_idx, scores_rel in zip(obj_indices, scores_rels):
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for ri in sorted_scores_rel[:num_top_rel_with]:
                triplets_pred_with.append((verb_idx.item(), obj_idx.item(), ri.item()))
                scores_pred_with.append(scores_rel[ri].item())

            for ri in sorted_scores_rel[:ceil(max(list_k)/num_obj)]:
                triplets_pred_no.append((verb_idx.item(), obj_idx.item(), ri.item()))
                scores_pred_no.append(scores_rel[ri].item())

        # make triplets for sgcls
        triplets_sg_with = []
        scores_sg_with = []
        triplets_sg_no = []
        scores_sg_no = []
        num_top_obj_with = ceil(max(list_k)/(num_top_rel_with*num_obj))
        num_top_obj_no = ceil(max(list_k)/(num_top_rel_no*num_obj))
        for scores_obj, scores_rel in zip(scores_objs, scores_rels):
            sorted_scores_obj = scores_obj.argsort(descending=True)
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for oi in sorted_scores_obj[:num_top_obj_with]:
                for ri in sorted_scores_rel[:num_top_rel_with]:
                    triplets_sg_with.append((verb_idx.item(), oi.item(), ri.item()))
                    scores_sg_with.append((scores_obj[oi]+scores_rel[ri]).item())
            for oi in sorted_scores_obj[:num_top_obj_no]:
                for ri in sorted_scores_rel[:num_top_rel_no]:
                    triplets_sg_no.append((verb_idx.item(), oi.item(), ri.item()))
                    scores_sg_no.append((scores_obj[oi]+scores_rel[ri]).item())

        # make triplets for easgcls
        triplets_easg_with = []
        scores_easg_with = []
        triplets_easg_no = []
        scores_easg_no = []
        num_top_obj_with = ceil(max(list_k)/(num_top_verb*num_top_rel_with*num_obj))
        num_top_obj_no = ceil(max(list_k)/(num_top_verb*num_top_rel_no*num_obj))
        for vi in scores_verb.argsort(descending=True)[:num_top_verb]:
            for scores_obj, scores_rel in zip(scores_objs, scores_rels):
                sorted_scores_obj = scores_obj.argsort(descending=True)
                sorted_scores_rel = scores_rel.argsort(descending=True)
                for oi in sorted_scores_obj[:num_top_obj_with]:
                    for ri in sorted_scores_rel[:num_top_rel_with]:
                        triplets_easg_with.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_with.append((scores_verb[vi]+scores_obj[oi]+scores_rel[ri]).item())
                for oi in sorted_scores_obj[:num_top_obj_no]:
                    for ri in sorted_scores_rel[:num_top_rel_no]:
                        triplets_easg_no.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_no.append((scores_verb[vi]+scores_obj[oi]+scores_rel[ri]).item())

        triplets_pred_with = torch.tensor(triplets_pred_with, dtype=torch.long)
        triplets_pred_no = torch.tensor(triplets_pred_no, dtype=torch.long)
        triplets_sg_with = torch.tensor(triplets_sg_with, dtype=torch.long)
        triplets_sg_no = torch.tensor(triplets_sg_no, dtype=torch.long)
        triplets_easg_with = torch.tensor(triplets_easg_with, dtype=torch.long)
        triplets_easg_no = torch.tensor(triplets_easg_no, dtype=torch.long)

        # sort the triplets using the averaged scores
        triplets_pred_with = triplets_pred_with[torch.argsort(torch.tensor(scores_pred_with), descending=True)].to(device)
        triplets_pred_no = triplets_pred_no[torch.argsort(torch.tensor(scores_pred_no), descending=True)].to(device)
        triplets_sg_with = triplets_sg_with[torch.argsort(torch.tensor(scores_sg_with), descending=True)].to(device)
        triplets_sg_no = triplets_sg_no[torch.argsort(torch.tensor(scores_sg_no), descending=True)].to(device)
        triplets_easg_with = triplets_easg_with[torch.argsort(torch.tensor(scores_easg_with), descending=True)].to(device)
        triplets_easg_no = triplets_easg_no[torch.argsort(torch.tensor(scores_easg_no), descending=True)].to(device)

        out_to_gt_pred_with = intersect_2d(triplets_gt, triplets_pred_with)
        out_to_gt_pred_no = intersect_2d(triplets_gt, triplets_pred_no)
        out_to_gt_sg_with = intersect_2d(triplets_gt, triplets_sg_with)
        out_to_gt_sg_no = intersect_2d(triplets_gt, triplets_sg_no)
        out_to_gt_easg_with = intersect_2d(triplets_gt, triplets_easg_with)
        out_to_gt_easg_no = intersect_2d(triplets_gt, triplets_easg_no)

        ## chech the mistakes 
        # s = out_to_gt_pred_with.sum(axis=1) # n_oggx1
        # print(s)
        # for el in s:
        #     if el == 0:
        #         print(f'objects: {obj_indices}')
        #         print(out_to_gt_pred_with)
        #         print(triplets_pred_with)
        #         print(triplets_gt)
        #         print(idx)

        num_gt = triplets_gt.shape[0]
        for k in list_k:
            recall_predcls_with[k].append(out_to_gt_pred_with[:, :k].any(dim=1).sum().item() / num_gt)
            recall_predcls_no[k].append(out_to_gt_pred_no[:, :k].any(dim=1).sum().item() / num_gt)
            recall_sgcls_with[k].append(out_to_gt_sg_with[:, :k].any(dim=1).sum().item() / num_gt)
            recall_sgcls_no[k].append(out_to_gt_sg_no[:, :k].any(dim=1).sum().item() / num_gt)
            recall_easgcls_with[k].append(out_to_gt_easg_with[:, :k].any(dim=1).sum().item() / num_gt)
            recall_easgcls_no[k].append(out_to_gt_easg_no[:, :k].any(dim=1).sum().item() / num_gt)

    for k in list_k:
        recall_predcls_with[k] = sum(recall_predcls_with[k]) / len(recall_predcls_with[k])*100
        recall_predcls_no[k] = sum(recall_predcls_no[k]) / len(recall_predcls_no[k])*100
        recall_sgcls_with[k] = sum(recall_sgcls_with[k]) / len(recall_sgcls_with[k])*100
        recall_sgcls_no[k] = sum(recall_sgcls_no[k]) / len(recall_sgcls_no[k])*100
        recall_easgcls_with[k] = sum(recall_easgcls_with[k]) / len(recall_easgcls_with[k])*100
        recall_easgcls_no[k] = sum(recall_easgcls_no[k]) / len(recall_easgcls_no[k])*100

    if dump_output:
        with open('easg_verb_output', 'wb') as fp:
            pickle.dump(verbs_predictions,fp)
    
    return recall_predcls_with, recall_predcls_no, recall_sgcls_with, recall_sgcls_no, recall_easgcls_with, recall_easgcls_no

def main():
    # GET EVALUATION DATASET
    logger = logging.getLogger()

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

    dataset_original = EASGData(path_annts, path_data, args.partition, verbs, objs, rels)
    dataset = myEASGDataset(dataset_original)

    # LOAD THE MODEL
    if args.edge_criterion == 'mean':
        edge_criterion = 'mean'
    elif args.edge_criterion == 'max':
        edge_criterion = 'max'
    elif args.edge_criterion == 'conc':
        edge_criterion = 'conc'
    else:
        raise Exception('Wrong edge criterion')
    
    obj_dim = 1024                  # original object dimension
    verb_dim = 2304                 # original verb dimension
    device = 'cuda' if cuda.is_available() else print('CUDA NOT AVAILABLE')

    model = EASGClassifier(obj_dim, verb_dim, 
                            num_rels, num_verbs, num_objs, 
                            args.hidden_proj_dim, args.proj_dim, args.hidden_dim, args.output_dim, 
                            device, args.dropout_prob, edge_criterion, args.graph_type)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    recall_predcls_with, recall_predcls_no, recall_sgcls_with, recall_sgcls_no, recall_easgcls_with, recall_easgcls_no = evaluation(dataset, model, device)
    print(f'with: [({recall_predcls_with[10]:.2f}, {recall_predcls_with[20]:.2f}, {recall_predcls_with[50]:.2f}), ({recall_sgcls_with[10]:.2f}, {recall_sgcls_with[20]:.2f}, {recall_sgcls_with[50]:.2f}), ({recall_easgcls_with[10]:.2f}, {recall_easgcls_with[20]:.2f}, {recall_easgcls_with[50]:.2f})], no: [({recall_predcls_no[10]:.2f}, {recall_predcls_no[20]:.2f}, {recall_predcls_no[50]:.2f}), ({recall_sgcls_no[10]:.2f}, {recall_sgcls_no[20]:.2f}, {recall_sgcls_no[50]:.2f}), ({recall_easgcls_no[10]:.2f}, {recall_easgcls_no[20]:.2f}, {recall_easgcls_no[50]:.2f})]')
    logger.info(f'with: [({recall_predcls_with[10]:.2f}, {recall_predcls_with[20]:.2f}, {recall_predcls_with[50]:.2f}), ({recall_sgcls_with[10]:.2f}, {recall_sgcls_with[20]:.2f}, {recall_sgcls_with[50]:.2f}), ({recall_easgcls_with[10]:.2f}, {recall_easgcls_with[20]:.2f}, {recall_easgcls_with[50]:.2f})], no: [({recall_predcls_no[10]:.2f}, {recall_predcls_no[20]:.2f}, {recall_predcls_no[50]:.2f}), ({recall_sgcls_no[10]:.2f}, {recall_sgcls_no[20]:.2f}, {recall_sgcls_no[50]:.2f}), ({recall_easgcls_no[10]:.2f}, {recall_easgcls_no[20]:.2f}, {recall_easgcls_no[50]:.2f})]')

if __name__ == "__main__":
    main()