from run_easg import EASGData
from pathlib import Path
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

class myEASGDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def get_object_indices(self, idx):
        data_dict = self.data_list[idx]
        obj_indices = data_dict['obj_indices']
        return obj_indices
    
    def get_verb_index(self, idx):
        data_dict = self.data_list[idx]
        verb_idx = data_dict['verb_idx']
        return verb_idx
    
    def get_original_triplets(self, idx):
        ''' this contain the original triplets with the indexing that uses
        the two different objects and verb files'''
        data_dict = self.data_list[idx]
        triplets = data_dict['triplets']
        return triplets
    
    def get_rels(self,idx):
        data_dict = self.data_list[idx]
        rels_vecs = data_dict['rels_vecs']
        return rels_vecs
    
    def triplets2edge_index(self, triplets):
        ''' 
            Create edge_index tensor starting from the original triplets.
            1. Get a temporary edge_index by selecting the first two columns of triplets
                and transposing the result
            2. Create a list of the nodes present (without repeating elements)
            3. Replace the indices in temporary edge index with the indices starting from 0 
                that we get from the list of unique nodes
            The result is an edge_index with indices going from 0 to the number of nodes 
            (verb+objects) of the graph. If we have two nodes linked by two or more relationships
            the edge index will have repeating columns.
        '''
        edge_index_temp = triplets[:,:2].t().contiguous()
        # Objects go from 0 to 390, verbs go from 0 to 197. It could happen that verb 5 is connected
        # to object 5 and I would have an edge index that goes from 5 to 5 when actually the nodes are 
        # different. To avoid this situation I add to the second row of the edge index the number of verbs
        # which is 198. This way it's like all nodes (verbs+objects) are indexed consecutively.
        # We can do it this way because we know that the first line always corresponds to a verb, 
        # without this knowledge this should be adapted
        edge_index_temp[1] = edge_index_temp[1]+198
        edge_index = torch.zeros(edge_index_temp.size(), dtype = torch.int64)
        nodes = torch.unique_consecutive(edge_index_temp.flatten()) # DO NOT SORT THE ELEMENTS, YOU WILL LOSE THE ORDER OF rels_vecs
        nodes = nodes.tolist()
        for i,row in enumerate(edge_index_temp):
            for j, el in enumerate(row):
                index = nodes.index(el)
                edge_index[i][j] = index
        return torch.unique_consecutive(edge_index, dim=1)
        # NB: given that we summed 198 to the object indices, the first line of edge_index
        # will always be zero in case of verb-object relationship

    def __getitem__(self, idx):
        # Extract data from the dictionary
        data_dict = self.data_list[idx]
        clip_features = data_dict['clip_feat']
        obj_feats = data_dict['obj_feats']
        triplets = data_dict['triplets']
        rels = data_dict['rels_vecs']
        verb_idx = data_dict['verb_idx']
        obj_indices = data_dict['obj_indices']

        # Concatenate clip and object features, pad the object features to match clip features
        clip_features = clip_features.unsqueeze(0)
        obj_feats = torch.cat([obj_feats, torch.zeros(obj_feats.size(0), clip_features.size(1)-obj_feats.size(1))], dim=1)
        x = torch.cat([clip_features, obj_feats], dim=0)

        # Create edge index tensor
        edge_index = self.triplets2edge_index(triplets)

        # Create target tensors
        y = (rels, verb_idx, obj_indices)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        return data