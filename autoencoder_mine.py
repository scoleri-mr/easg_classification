import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv
from torch_geometric.nn import global_max_pool

class LinearProjection(nn.Module):
    def __init__(self, verb_dim, obj_dim, hidden_projection_dim, projection_dim, dropout_prob):
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
        
        self.mlp_object = nn.Sequential(
            nn.Linear(obj_dim, hidden_projection_dim),
            nn.LayerNorm(hidden_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_projection_dim, projection_dim),
        )
        
        self.mlp_verb = nn.Sequential(
            nn.Linear(verb_dim, hidden_projection_dim),
            nn.LayerNorm(hidden_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_projection_dim, projection_dim),
        )

    def forward(self, batch):
        """
        process with mlps, returns processed verb_feat and obj_feat
        """
        out, mask = tg.utils.to_dense_batch(batch.x,batch.batch)  # [bs,max_nodes,2304], [bs,max_nodes]
        bs, max_nodes = out.size(0), out.size(1)
        # re-arrange features + remove feature padding: first elem. is verb node, the others are objs
        verb_feat = out[:,0,:self.verb_dim].unsqueeze(1)  # [bs, 1, verb_feats_dim]
        obj_feat = out[:,1:,:self.obj_dim]  # [bs, max_nodes-1, object_feats_dim]

        assert verb_feat.ndim == 3 and obj_feat.ndim == 3, "check input shapes to LinearProj"
        verb_feat = self.mlp_verb(verb_feat)  # [bs,1,projection_dim]
        obj_feat = self.mlp_object(obj_feat)  # [bs,max_nodes-1,projection_dim]

        nodes_features = torch.cat([verb_feat, obj_feat], dim=1)  # [bs,max_nodes,projection_dim]
        nodes_features = nodes_features[mask]
        batch.x = nodes_features
        return batch
    
class myGNN(nn.Module):
    ''' 
        apply two gnn layers ('gcn', 'sage' or 'gat') to the graph and get the edge features by
        simply returning the elementwise max or mean between the two nodes that 
        form the edge    
    '''
    def __init__(self, layer_type, input_dim, hidden_dim, output_dim, dropout_prob):
        super().__init__()
        self.output_dim = output_dim
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
        elif layer_type=='gin':
            self.conv1 = GINConv(nn.Sequential(
                nn.Linear(input_dim, hidden_dim)
            ))
            self.conv2 = GINConv(nn.Sequential(
                nn.Linear(hidden_dim, output_dim)
            ))
        else:
            raise Exception('Wrong graph layer type')
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.adaptive_max = nn.AdaptiveMaxPool1d(output_dim)

    def forward(self, batch):
        nodes_features = self.dropout(self.relu(self.conv1(batch.x, batch.edge_index)))
        nodes_features = self.dropout(self.relu(self.conv2(nodes_features, batch.edge_index)))
        batch.x = nodes_features
        return batch
    
class EASGEncoder(nn.Module): 
    def __init__(
        self, object_feats_dim, verb_feats_dim, 
        hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
        dropout_prob=0.2, graph_type='gat'
        ):
        super().__init__()
        self.object_feats_dim = object_feats_dim
        self.verb_feats_dim = verb_feats_dim
        self.projection_dim = projection_dim
        self.verb_obj_proj =  LinearProjection(
            verb_dim=verb_feats_dim, obj_dim=object_feats_dim, 
            hidden_projection_dim=hidden_projection_dim, projection_dim=projection_dim, 
            dropout_prob=dropout_prob
            )
        self.graph_type = graph_type
        self.gnn = myGNN(graph_type, projection_dim, hidden_dim, output_dim, dropout_prob)

    def encode_graph(self, batch):
        batch = self.verb_obj_proj(batch)
        batch = self.gnn(batch)
        graphs_latents = global_max_pool(batch.x, batch.batch)
        return batch.x, graphs_latents
    
    def forward(self, batch):
        return self.encode_graph(batch)