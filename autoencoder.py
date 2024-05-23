import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv, PNAConv, global_max_pool
from torch_geometric.nn import global_add_pool
from models import LinearProjection
import torch_geometric as tg

class myGNN(nn.Module):
    ''' 
        Differs from myGNN in models.py because it doesn't generate edge_features
        apply two gnn layers ('gcn', 'sage', 'gat' or 'gin') to the graph and get the edge features by
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

    def forward(self, nodes_features, edge_index):
        nodes_features = self.dropout(self.relu(self.conv1(nodes_features, edge_index)))
        nodes_features = self.dropout(self.relu(self.conv2(nodes_features, edge_index)))
        return nodes_features
    
# Encoder 
class EASGEncoder(nn.Module):
    def __init__(self, object_feats_dim, verb_feats_dim,
                 num_rels, num_verbs, num_objs, hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
                 dropout_prob=0.2, graph_type='gat', pool_type='add'):
        super().__init__()
        self.object_feats_dim = object_feats_dim
        self.verb_feats_dim = verb_feats_dim
        self.projection_dim = projection_dim
        self.verb_obj_proj =  LinearProjection(
            verb_dim=verb_feats_dim, obj_dim=object_feats_dim, hidden_projection_dim=hidden_projection_dim, 
            projection_dim=projection_dim, dropout_prob=dropout_prob)
        self.graph_type = graph_type
        self.pool_type = pool_type
        self.gnn = myGNN(graph_type, projection_dim, hidden_dim, output_dim, dropout_prob)

    def encode_graph(self, batch):
        """ forward takes whole batch now """
        out, mask = tg.utils.to_dense_batch(batch.x,batch.batch)  # [bs,max_nodes,2304], [bs,max_nodes]        
        # re-arrange features + remove feature padding
        # first elem. is verb node
        verb_feat = out[:,0,:self.verb_feats_dim].unsqueeze(1)  # [bs, 1, verb_feats_dim]
        # after first eleme at each batch item we've objs feats
        # "obj_feat" contains also object nodes padded with zeros... we don't mind as we filter them later
        obj_feat = out[:,1:,:self.object_feats_dim]  # [bs, max_nodes-1, object_feats_dim]

        # process with mlps
        verb_feat, obj_feat = self.verb_obj_proj(verb_feat, obj_feat)  # [bs,1,proj_dim], [bs,max_nodes-1,proj_dim]
        
        # put again verb and objs nodes
        nodes_features = torch.cat([verb_feat, obj_feat], dim=1)  # [bs,max_nodes,projection_dim]
        # to PyG list + remove padding nodes
        nodes_features = nodes_features[mask]
        
        # edge index are kept the same
        edge_index = batch.edge_index
        
        nodes_features = self.gnn(nodes_features, edge_index)

        if self.pool_type == 'add':
            graphs_latents = global_add_pool(nodes_features, batch.batch)
        elif self.pool_type == 'max':
            graphs_latents = global_max_pool(nodes_features, batch.batch)
        else:
            raise Exception('Wrong pooling type')
        return nodes_features, graphs_latents
    
    def forward(self, batch):
        nodes_features, graphs_latents = self.encode_graph(batch)
        return nodes_features, graphs_latents

class EASGDecoder(nn.Module): 
    def __init__(
        self, num_rels, num_verbs, num_objs, input_dim, hidden_dim, 
        dropout_prob=0.2
        ):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # this part is for classifying the verb starting from the graph latent code
        self.verb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_verbs),
        )
        
        # this part is for classifying the object-verb relationship from the graph latent code
        self.rel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_objs*64),
        )
        self.rel_head = nn.Conv1d(in_channels=64, out_channels=num_rels, kernel_size=1)
        
    def forward(self, codes):
        """ decoder takes a latent code for each graph """
        bs = codes.size(0)
        # upsample features - common for verb and obj-verb relationships
        shared_repr = self.shared_mlp(codes)
        
        #########
        # verb classification
        verb_cls = self.verb_head(shared_repr)  # [bs, num_verbs]
        #########
        
        #########
        # obj-verb rel classification
        relationships = self.rel_mlp(shared_repr)  # [bs, num_objs*64]
        relationships = relationships.view(bs, -1, 64) #  [bs, num_objs, 64]
        relationships = relationships.permute(0,2,1) #  [bs, 64, num_objs]
        relationships_cls = self.rel_head(relationships) #  [bs, num_rel, num_objs]
        relationships_cls = relationships_cls.permute(0,2,1) #  [bs, num_objs, num_rel]

        ### ADDING THIS RESHAPE  ?????
        relationships_cls = relationships_cls.reshape(bs * 391, 14)  # [bs*num_objs, num_rel]
        #########
        
        return verb_cls, relationships_cls
    
class EASGAutoEncoder(nn.Module): 
    def __init__(
        self, object_feats_dim, verb_feats_dim, num_rels, num_verbs, num_objs, hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
        dropout_prob=0.2, graph_type='gat'
        ):
        super().__init__()
        
        self.encoder = EASGEncoder(
            object_feats_dim, verb_feats_dim, num_rels, num_verbs, num_objs, hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
            dropout_prob, graph_type
        )
        self.decoder = EASGDecoder(
            num_rels, num_verbs, num_objs, output_dim, output_dim*2, dropout_prob
        )
    
    def encode(self, batch):
        _, graphs_latents = self.encoder(batch)
        return graphs_latents
    
    def decode(self, graphs_latents):
        verb_cls, relationships_cls = self.decoder(graphs_latents)
        return verb_cls, relationships_cls
    
    def forward(self, batch):
        # encode 
        graphs_latents = self.encode(batch)
        
        # decode
        verb_logits, obj_verb_rel_logits = self.decode(graphs_latents)
        return verb_logits, obj_verb_rel_logits