import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from run_easg import EASGData
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import cuda
from torch.optim import Adam
import wandb
from dataset import myEASGDataset

class LinearProjection(nn.Module):
    def __init__(self, verb_dim, obj_dim, hidden_projection_dim, projection_dim, device):
        '''
            Originally, object features were 1024 and verb features were 2304.
            Both verbs and objects need to be considered nodes so objects are padded in the dataset.
            We don't want to send padded nodes to the gnn so we handle this here:
            we perform a linear projection of objects and verbs removing the padding.
        '''
        super().__init__()
        self.projection_dim = projection_dim
        self.obj_dim = obj_dim
        self.verb_dim = verb_dim
        self.device = device
        self.verb_fc1 = nn.Linear(verb_dim, hidden_projection_dim)
        self.verb_fc2 = nn.Linear(hidden_projection_dim, projection_dim)
        self.obj_fc1 = nn.Linear(obj_dim, hidden_projection_dim)
        self.obj_fc2 = nn.Linear(hidden_projection_dim, projection_dim)

    def forward(self, x):
        new_x = torch.zeros([x.size(0), self.projection_dim])
        new_x = new_x.to(self.device)
        for i,_ in enumerate(x):
            if i==0: 
                # in edge index the first node is always the verb: take all x[0]
                temp = F.relu(self.verb_fc1(x[i]))
                new_x[i] = self.verb_fc2(temp)
            else:
                # the other nodes are objects: take x[:object_dim]
                temp = F.relu(self.obj_fc1(x[i][:self.obj_dim]))
                new_x[i] = self.obj_fc2(temp)
        return new_x
    
class myGCN(nn.Module):
    ''' 
        apply two gcn layers to the graph and get the edge features by
        simply returning the elementwise max between the two nodes that 
        form the edge    
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, device, edge_creation='mean'):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self.edge_creation = edge_creation
    
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))        
        x = F.relu(self.conv2(x, edge_index))
        edge_features = self.compute_edge_features(x, edge_index, self.output_dim)
        return edge_features
    
    def compute_edge_features(self, x, edge_index, edge_dim):
        edge_features = torch.zeros([edge_index.size(1), edge_dim])
        edge_features = edge_features.to(self.device)
        for i in range(edge_index.size(1)):
            if self.edge_creation == 'max':
                edge_features[i] = torch.max(x[edge_index[0][i]], x[edge_index[1][i]])
            elif self.edge_creation == 'mean':
                edge_features[i] = (x[edge_index[0][i]] + x[edge_index[1][i]])/2
        return edge_features

class myClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits

class EdgeClassifier(nn.Module):
    def __init__(self, object_feats_dim, verb_feats_dim, projection_dim,
                 hidden_dim, output_dim, hidden_projection_dim, num_relationships, device = 'cuda'):
        super().__init__()
        self.projection_dim = projection_dim
        
        self.linear_projection = LinearProjection(verb_feats_dim, object_feats_dim, hidden_projection_dim, projection_dim, device)
        self.gcn = myGCN(projection_dim, hidden_dim, output_dim, device)
        self.cls = myClassifier(output_dim, num_relationships)

    def forward(self, x, edge_index):
        x = self.linear_projection(x)
        x = self.gcn(x, edge_index)
        x = self.cls(x)
        return x