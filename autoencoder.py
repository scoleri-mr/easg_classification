import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv
from torch_geometric.nn import global_max_pool
from torchvision.ops.focal_loss import sigmoid_focal_loss


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
    def __init__(self, layer_type, input_dim, hidden_dim, output_dim, dropout_prob, num_heads=8):
        super().__init__()
        self.output_dim = output_dim
        self.layer_type = layer_type
        self.num_heads = num_heads

        if layer_type=='gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif layer_type=='sage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        elif layer_type=='gat':
            self.conv1 = GATv2Conv(input_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout_prob)
            self.conv2 = GATv2Conv(hidden_dim, output_dim, heads=1, concat=True, dropout=dropout_prob)
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
    
class EASGDecoder(nn.Module): 
    def __init__(
        self, num_rels, num_verbs, num_objs, input_dim, hidden_dim, 
        dropout_prob=0.2, separate=False
        ):
        super().__init__()
        self.separate = separate
        if separate: 
            self.shared_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            # verb cls starting from latent graph
            self.verb_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_verbs),
            )

            self.rel_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_objs*64),
            )

        else: 
            self.shared_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            
            # verb cls starting from latent graph
            self.verb_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_verbs),
            )
            
            # rels cls starting from latent graph
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
        if not self.separate: shared_repr = self.shared_mlp(codes)
        
        # verb classification
        if self.separate:
            verb_logits = self.verb_head(codes)  # [bs, num_verbs]
        else: verb_logits = self.verb_head(shared_repr)  # [bs, num_verbs]
        
        # obj-verb rel classification
        if self.separate:
            relationships = self.rel_mlp(codes)
        else: relationships = self.rel_mlp(shared_repr)  # [bs, num_objs*64]
        relationships = relationships.view(bs, -1, 64) #  [bs, num_objs, 64]
        relationships = relationships.permute(0,2,1) #  [bs, 64, num_objs]
        relationships_logits = self.rel_head(relationships) #  [bs, num_rel, num_objs]
        relationships_logits = relationships_logits.permute(0,2,1) #  [bs, num_objs, num_rel]

        return verb_logits, relationships_logits

class EASGAutoEncoder(nn.Module): 
    def __init__(   self, object_feats_dim, verb_feats_dim, 
                    num_rels, num_verbs, num_objs, 
                    hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
                    dropout_prob=0.2, graph_type='gat', use_focal_loss=False, separate=False):
        super().__init__()
        
        self.encoder = EASGEncoder(
            object_feats_dim, verb_feats_dim, 
            hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
            dropout_prob, graph_type
        )
        self.decoder = EASGDecoder(num_rels, num_verbs, num_objs, 
                                    output_dim, output_dim*2, dropout_prob, separate=separate)
        self.focal_loss_verb = MultiClassFocalLoss()
        self.use_focal_loss = use_focal_loss
        if separate:
            print("Using separate mlp for verb and rels, removing shared mlp...")

        if self.use_focal_loss:
            print("Using ae with focal loss...")        
        
    def forward(self, batch):
        # encode 
        _, graphs_latents = self.encoder.encode_graph(batch)
        
        # decode
        verb_logits, obj_verb_rel_logits = self.decoder(graphs_latents)
        return verb_logits, obj_verb_rel_logits

    def loss_functions_ae(self, verb_gt, rels_gt, verb_logits, relationship_logits):
        relationship_logits = relationship_logits.contiguous().view(-1, 14) 
        rels_gt = rels_gt.view(-1, 14)
        if self.use_focal_loss:
            loss_verb = self.focal_loss_verb(verb_logits, verb_gt)
            loss_rel = sigmoid_focal_loss(relationship_logits, rels_gt, reduction="mean")
        else:
            loss_verb = F.cross_entropy(input=verb_logits, target=verb_gt)
            loss_rel = F.binary_cross_entropy_with_logits(input=relationship_logits, target=rels_gt)
        return loss_verb, loss_rel
    
class EASGvae(nn.Module):
    def __init__(   self, object_feats_dim, verb_feats_dim, 
                    num_rels, num_verbs, num_objs, 
                    hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
                    kld_type, dropout_prob=0.2, graph_type='gcn', use_focal_loss=False, eps=1., separate=True,
                    class_14_weight_factor=0.01, balance_losses=False):
        super(EASGvae, self).__init__()
        self.kld_type = kld_type
        self.encoder = EASGEncoder(object_feats_dim, verb_feats_dim, 
                                   hidden_projection_dim, projection_dim, hidden_dim, output_dim, 
                                   dropout_prob, graph_type)
        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_logvar = nn.Linear(output_dim, output_dim)
        self.decoder = EASGDecoder(num_rels, num_verbs, num_objs, 
                                   output_dim, hidden_dim, dropout_prob, separate)
        self.focal_loss_verb = MultiClassFocalLoss()
        self.use_focal_loss = use_focal_loss
        self.eps = eps
        self.separate = separate
        self.class_14_weight_factor = class_14_weight_factor
        self.balance_losses = balance_losses

        if self.balance_losses:
            print("Using balanced losses...")
            self.sigma_verb = nn.Parameter(torch.tensor(1.0))
            self.sigma_rel = nn.Parameter(torch.tensor(1.0))
            self.sigma_kld = nn.Parameter(torch.tensor(1.0))

        if self.separate:
            print("Using separate mlp for verb and rels, removing shared mlp...")

        if self.use_focal_loss:
            print("Using vae with focal loss...")

    def forward(self, batch):
        _, graphs_latents = self.encoder(batch)
        mu = self.fc_mu(graphs_latents)
        logvar = self.fc_logvar(graphs_latents)
        graphs_latents = self.reparameterize(mu, logvar, self.eps)
        verb_logits, relationships_logits = self.decoder(graphs_latents)
        return verb_logits, relationships_logits, mu, logvar
    
    #### the loss function in neural graph generator repeats parts of the forward, I removed those parts
    #### with respect to a traditional VAE instead of having an l1 type loss we use a CE and BCE that are summed to the kld
    def loss_functions(self, verb_gt, rels_gt, verb_logits, relationship_logits, mu, logvar):
        relationship_logits = relationship_logits.contiguous().view(-1, 14) 
        rels_gt = rels_gt.view(-1, 14)
        if self.use_focal_loss:
            loss_verb = self.focal_loss_verb(verb_logits, verb_gt)
            loss_rel = sigmoid_focal_loss(relationship_logits, rels_gt, reduction="mean")
            # weights = torch.cat((torch.ones(13), torch.tensor([0.01])))
            # loss_rel = F.binary_cross_entropy_with_logits(weight=weights, input=relationship_logits, target=rels_gt)
        else:
            # still use the focal loss for verbs
            loss_verb = self.focal_loss_verb(verb_logits, verb_gt)
            
            # Binary cross-entropy with class weighting for relationships
            # Here we assign a lower weight to class 14 (index 13) by using the `class_14_weight_factor`
            weights = torch.ones(relationship_logits.size(1), device=relationship_logits.device)
            weights[13] = self.class_14_weight_factor
            
            loss_rel = F.binary_cross_entropy_with_logits(
                input=relationship_logits, 
                target=rels_gt,
                weight=weights
            )
        if self.kld_type == 'original': # performs kld summing all together for the batch
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        elif self.kld_type == 'mean':   # performs separate kld for each sample and then average them
            kld =  torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
        return loss_verb, loss_rel, kld
        
    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, batch):
        _, graphs_latents = self.encoder(batch)
        mu = self.fc_mu(graphs_latents)
        logvar = self.fc_logvar(graphs_latents)
        graphs_latents = self.reparameterize(mu, logvar)
        return graphs_latents
    
    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=-1)
        logpt = logpt.gather(1, targets.view(-1, 1))
        logpt = logpt.view(-1)
        pt = logpt.exp()

        focal_loss = -((1 - pt) ** self.gamma) * logpt

        if self.alpha >= 0:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    