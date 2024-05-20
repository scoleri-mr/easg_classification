import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv, PNAConv
from torch_geometric.nn import global_add_pool

class LinearProjection(nn.Module):
    def __init__(self, verb_dim, obj_dim, hidden_projection_dim, projection_dim, device, dropout_prob):
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
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, batch):
        batch_indices = batch.batch
        batch_unique = torch.unique(batch_indices)
        new_batch = []
        for el in batch_unique:
            select = batch.x[batch_indices==el]
            verb = select[0]
            objs = select[1:]

            # handle the verb: take all the features
            temp = self.dropout(self.relu(self.l_norm(self.verb_fc1(verb))))
            new_batch.append(self.l_norm2(self.verb_fc2(temp)))

            # handle the objects: take x[:object_dim]
            for obj in objs:
                temp = self.dropout(self.relu(self.l_norm(self.obj_fc1(obj[:self.obj_dim]))))
                new_batch.append(self.l_norm2(self.obj_fc2(temp)))

        new_batch = torch.stack(new_batch, dim=0)
        return new_batch

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out
    
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
        elif layer_type=='gin':
            self.conv1 = GINConv(input_dim, hidden_dim)
            self.conv2 = GINConv(hidden_dim, output_dim)
        else:
            raise Exception('Wrong graph layer type')
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.adaptive_max = nn.AdaptiveMaxPool1d(output_dim)

    def forward(self, nodes_features, edge_index):
        nodes_features = self.dropout(self.relu(self.conv1(nodes_features, edge_index)))
        nodes_features = self.dropout(self.relu(self.conv2(nodes_features, edge_index)))
        return nodes_features
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj

# Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(AutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        return x_g

    def decode(self, x_g):
        adj = self.decoder(x_g)
        return adj

    def loss_function(self, data):
        x_g  = self.encoder(data)
        adj = self.decoder(x_g)
        A = data.A[:,:,:,0]
        return F.l1_loss(adj, data.A)