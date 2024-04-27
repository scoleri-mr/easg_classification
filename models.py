import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv

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
        self.l_norm = nn.LayerNorm(hidden_projection_dim)
        self.relu = nn.ReLU()
        self.l_norm2 = nn.LayerNorm(projection_dim)

    def forward(self, x):
        new_x = torch.zeros([x.size(0), self.projection_dim])
        new_x = new_x.to(self.device)
        for i,_ in enumerate(x):
            if i==0: 
                # in edge index the first node is always the verb: take all x[0]
                temp = self.relu(self.l_norm(self.verb_fc1(x[i])))
                new_x[i] = self.l_norm2(self.verb_fc2(temp))
            else:
                # the other nodes are objects: take x[:object_dim]
                temp = self.relu(self.l_norm(self.obj_fc1(x[i][:self.obj_dim])))
                new_x[i] = self.l_norm2(self.obj_fc2(temp))
        return new_x
    
class myGNN(nn.Module):
    ''' 
        apply two gnn layers ('gcn', 'sage' or 'gat') to the graph and get the edge features by
        simply returning the elementwise max or mean between the two nodes that 
        form the edge    
    '''
    def __init__(self, layer_type, input_dim, hidden_dim, output_dim, dropout_prob, edge_creation, device):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self.edge_creation = edge_creation
        self.layer_type = layer_type

        if layer_type=='gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif layer_type=='sage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        elif layer_type=='gat':
            self.conv1 = GATv2Conv(input_dim, hidden_dim)
            self.conv2 = GATv2Conv(hidden_dim, output_dim)
        else:
            raise Exception('Wrong graph layer type')
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, nodes_features, edge_index):
        nodes_features = self.dropout(self.relu(self.conv1(nodes_features, edge_index)))
        nodes_features = self.dropout(self.relu(self.conv2(nodes_features, edge_index)))
        edge_features = self.compute_edge_features(nodes_features, edge_index, self.output_dim)
        return nodes_features, edge_features
    
    def compute_edge_features(self, nodes_features, edge_index, edge_dim):
        edge_features = torch.zeros([edge_index.size(1), edge_dim])
        edge_features = edge_features.to(self.device)
        for i in range(edge_index.size(1)):
            if self.edge_creation == 'max':
                edge_features[i] = torch.max(nodes_features[edge_index[:, i]], dim=0).values
            elif self.edge_creation == 'mean':
                edge_features[i] = torch.mean(nodes_features[edge_index[:, i]], dim=0)
            elif self.edge_creation == 'conc':
                # NOT SUPPORTED YET
                m = torch.max(nodes_features[edge_index[:, i]], dim=0).values
                av = torch.mean(nodes_features[edge_index[:, i]], dim=0)
                edge_features[i] = torch.cat((m,av), dim=0)
        return edge_features

class myClassifier(nn.Module):
    def __init__(self, input_dim, num_rels, num_verbs, num_objs):
        super().__init__()
        self.fc_edges = nn.Linear(input_dim, num_rels)
        self.fc_verbs = nn.Linear(input_dim, num_verbs)
        self.fc_objs = nn.Linear(input_dim, num_objs)

    def forward(self, nodes_features, edge_features):
        logits_edges = self.fc_edges(edge_features) # n_edgesx13
        logits_verb = self.fc_verbs(nodes_features[0].unsqueeze(0)) # 1x198
        logits_objs = self.fc_objs(nodes_features[1:]) #n_oggx391
        return logits_edges, logits_verb, logits_objs 

class EASGClassifier(nn.Module): 
    def __init__(self, object_feats_dim, verb_feats_dim, 
                 num_rels, num_verbs, num_objs, 
                 hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
                 device = 'cuda', dropout_prob=0.2, edge_creation='mean', graph_type='gat'):
        super().__init__()
        self.projection_dim = projection_dim
        self.graph_type = graph_type
        
        self.linear_projection = LinearProjection(verb_feats_dim, object_feats_dim, hidden_projection_dim, projection_dim, device)
        self.gnn = myGNN(graph_type, projection_dim, hidden_dim, output_dim, dropout_prob, edge_creation, device)
        self.cls = myClassifier(output_dim, num_rels, num_verbs, num_objs)

    def forward(self, nodes_features, edge_index):
        nodes_features = self.linear_projection(nodes_features)
        nodes_features, edge_features = self.gnn(nodes_features, edge_index)
        logits_edges, logits_verb, logits_objs = self.cls(nodes_features, edge_features)
        return logits_edges, logits_verb, logits_objs