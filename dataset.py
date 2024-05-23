from pathlib import Path
import torch
from torch_geometric.data import Data, Dataset
import copy
import pickle

# original dataset
class EASGData(Dataset):
    def __init__(self, path_annts, path_data, split, verbs, objs, rels):
        self.path_annts = path_annts
        self.path_data = path_data
        self.split = split
        with open(path_annts / f'easg_{split}.pkl', 'rb') as f:
            annts = pickle.load(f)

        with open(path_data / f'roi_feats_{split}.pkl', 'rb') as f:
            roi_feats = pickle.load(f)

        clip_feats = torch.load(path_data / 'verb_features.pt')

        """
        graph:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['objs']: dict of obj_idx
                dict[obj_idx]: dict    
                    dict['obj_feat']: 1024-D ROI feature vector
                    dict['rels_vec']: multi-hot vector of relationships

        graph_batch:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['obj_indices']: batched version of obj_idx
            dict['obj_feats']: batched version of obj_feat
            dict['rels_vecs']: batched version of rels_vec
            dict['triplets']: all the triplets consisting of (verb, obj, rel)
        """
        graphs = []
        for graph_uid in annts:
            graph = {}
            for aid in annts[graph_uid]['annotations']: # cycle on all the annotations of the graph
                for i, annt in enumerate(annts[graph_uid]['annotations'][aid]):
                    verb_idx = verbs.index(annt['verb'])
                    if verb_idx not in graph:
                        graph[verb_idx] = {}
                        graph[verb_idx]['verb_idx'] = verb_idx
                        graph[verb_idx]['objs'] = {}

                    graph[verb_idx]['clip_feat'] = clip_feats[aid]

                    obj_idx = objs.index(annt['obj'])
                    if obj_idx not in graph[verb_idx]['objs']:
                        graph[verb_idx]['objs'][obj_idx] = {}
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.zeros((0, 1024), dtype=torch.float32)
                        graph[verb_idx]['objs'][obj_idx]['rels_vec'] = torch.zeros(len(rels), dtype=torch.float32)

                    rel_idx = rels.index(annt['rel'])
                    graph[verb_idx]['objs'][obj_idx]['rels_vec'][rel_idx] = 1

                    for frameType in roi_feats[graph_uid][aid][i]:
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.cat((graph[verb_idx]['objs'][obj_idx]['obj_feat'], roi_feats[graph_uid][aid][i][frameType]), dim=0)

            for verb_idx in graph:
                for obj_idx in graph[verb_idx]['objs']:
                    graph[verb_idx]['objs'][obj_idx]['obj_feat'] = graph[verb_idx]['objs'][obj_idx]['obj_feat'].mean(dim=0)

                graphs.append(graph[verb_idx])

        self.graphs = []
        for graph in graphs:
            graph_batch = {}
            verb_idx = graph['verb_idx']
            graph_batch['verb_idx'] = torch.tensor([verb_idx], dtype=torch.long)
            graph_batch['clip_feat'] = graph['clip_feat']
            graph_batch['obj_indices'] = torch.zeros(0, dtype=torch.long)
            graph_batch['obj_feats'] = torch.zeros((0, 1024), dtype=torch.float32)
            graph_batch['rels_vecs'] = torch.zeros((0, len(rels)), dtype=torch.float32)
            graph_batch['triplets'] = torch.zeros((0, 3), dtype=torch.long)

            for obj_idx in graph['objs']:
                graph_batch['obj_indices'] = torch.cat((graph_batch['obj_indices'], torch.tensor([obj_idx], dtype=torch.long)), dim=0)
                graph_batch['obj_feats'] = torch.cat((graph_batch['obj_feats'], graph['objs'][obj_idx]['obj_feat'].unsqueeze(0)), dim=0)

                rels_vec = graph['objs'][obj_idx]['rels_vec']
                graph_batch['rels_vecs'] = torch.cat((graph_batch['rels_vecs'], rels_vec.unsqueeze(0)), dim=0)

                triplets = []
                for rel_idx in torch.where(rels_vec)[0]:
                    triplets.append((verb_idx, obj_idx, rel_idx.item()))
                graph_batch['triplets'] = torch.cat((graph_batch['triplets'], torch.tensor(triplets, dtype=torch.long)), dim=0)

            self.graphs.append(graph_batch)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
    
    
class EASGDatasetAE(Dataset):
    def __init__(self, path_annts, path_data, split, verbs, objs, rels):
        self.whoami = "EASGDatasetAE"
        self.path_annts = path_annts
        self.path_data = path_data
        self.split = split
        self.num_objs = len(objs)
        self.num_verbs = len(verbs)
        self.num_rels = len(rels)
        self.verbs = verbs
        self.objs = objs
        self.rels = rels
        
        print(f"{self.whoami} - {split} - num_objs: {self.num_objs}, num_verbs: {self.num_verbs}, num_rels: {self.num_rels} ")
        with open(path_annts / f'easg_{split}.pkl', 'rb') as f:
            annts = pickle.load(f)

        with open(path_data / f'roi_feats_{split}.pkl', 'rb') as f:
            roi_feats = pickle.load(f)

        clip_feats = torch.load(path_data / 'verb_features.pt')

        """
        graph:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['objs']: dict of obj_idx
                dict[obj_idx]: dict    
                    dict['obj_feat']: 1024-D ROI feature vector
                    dict['rels_vec']: multi-hot vector of relationships

        graph_batch:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['obj_indices']: batched version of obj_idx
            dict['obj_feats']: batched version of obj_feat
            dict['rels_vecs']: batched version of rels_vec
            dict['triplets']: all the triplets consisting of (verb, obj, rel)
        """
        graphs = []
        for graph_uid in annts:
            graph = {}
            for aid in annts[graph_uid]['annotations']: # cycle on all the annotations of the graph
                for i, annt in enumerate(annts[graph_uid]['annotations'][aid]):
                    verb_idx = verbs.index(annt['verb'])
                    if verb_idx not in graph:
                        graph[verb_idx] = {}
                        graph[verb_idx]['verb_idx'] = verb_idx
                        graph[verb_idx]['objs'] = {}

                    graph[verb_idx]['clip_feat'] = clip_feats[aid]

                    obj_idx = objs.index(annt['obj'])
                    if obj_idx not in graph[verb_idx]['objs']:
                        graph[verb_idx]['objs'][obj_idx] = {}
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.zeros((0, 1024), dtype=torch.float32)
                        graph[verb_idx]['objs'][obj_idx]['rels_vec'] = torch.zeros(len(rels), dtype=torch.float32)

                    rel_idx = rels.index(annt['rel'])
                    graph[verb_idx]['objs'][obj_idx]['rels_vec'][rel_idx] = 1

                    for frameType in roi_feats[graph_uid][aid][i]:
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.cat((graph[verb_idx]['objs'][obj_idx]['obj_feat'], roi_feats[graph_uid][aid][i][frameType]), dim=0)

            for verb_idx in graph:
                for obj_idx in graph[verb_idx]['objs']:
                    graph[verb_idx]['objs'][obj_idx]['obj_feat'] = graph[verb_idx]['objs'][obj_idx]['obj_feat'].mean(dim=0)

                graphs.append(graph[verb_idx])

        self.graphs = []
        for graph in graphs:
            graph_batch = {}
            verb_idx = graph['verb_idx']
            graph_batch['verb_idx'] = torch.tensor([verb_idx], dtype=torch.long)
            graph_batch['clip_feat'] = graph['clip_feat']
            graph_batch['obj_indices'] = torch.zeros(0, dtype=torch.long)
            graph_batch['obj_feats'] = torch.zeros((0, 1024), dtype=torch.float32)
            graph_batch['rels_vecs'] = torch.zeros((0, len(rels)), dtype=torch.float32)
            graph_batch['triplets'] = torch.zeros((0, 3), dtype=torch.long)

            for obj_idx in graph['objs']:
                graph_batch['obj_indices'] = torch.cat((graph_batch['obj_indices'], torch.tensor([obj_idx], dtype=torch.long)), dim=0)
                graph_batch['obj_feats'] = torch.cat((graph_batch['obj_feats'], graph['objs'][obj_idx]['obj_feat'].unsqueeze(0)), dim=0)

                rels_vec = graph['objs'][obj_idx]['rels_vec']
                graph_batch['rels_vecs'] = torch.cat((graph_batch['rels_vecs'], rels_vec.unsqueeze(0)), dim=0)

                triplets = []
                for rel_idx in torch.where(rels_vec)[0]:
                    triplets.append((verb_idx, obj_idx, rel_idx.item()))
                graph_batch['triplets'] = torch.cat((graph_batch['triplets'], torch.tensor(triplets, dtype=torch.long)), dim=0)

            self.graphs.append(graph_batch)

    def __len__(self):
        return len(self.graphs)
    
    def get_verb_name(self, idx):
        verb_idx = self.get_verb_index(idx)
        return self.verbs[idx]
    
    def get_objs_name(self, idx):
        obj_indices = self.get_object_indices(idx)
        return self.objs[obj_indices]
    
    def get_rels_name(self, idx):
        rels_vecs = self.get_rels(idx)
        return [self.rels[el.argmax()] for el in rels_vecs]

    def get_object_indices(self, idx):
        data_dict = self.graphs[idx]
        obj_indices = data_dict['obj_indices']
        return obj_indices

    def get_verb_index(self, idx):
        data_dict = self.graphs[idx]
        verb_idx = data_dict['verb_idx']
        return verb_idx

    def get_original_triplets(self, idx):
        ''' this contain the original triplets with the indexing that uses
        the two different objects and verb files'''
        data_dict = self.graphs[idx]
        triplets = data_dict['triplets']
        return triplets

    def get_rels(self, idx):
        data_dict = self.graphs[idx]
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
        edge_index_temp = triplets[:, :2].t().contiguous()
        # Objects go from 0 to 390, verbs go from 0 to 197. It could happen that verb 5 is connected
        # to object 5 and I would have an edge index that goes from 5 to 5 when actually the nodes are 
        # different. To avoid this situation I add to the second row of the edge index the number of verbs
        # which is 198. This way it's like all nodes (verbs+objects) are indexed consecutively.
        # We can do it this way because we know that the first line always corresponds to a verb, 
        # without this knowledge this should be adapted
        edge_index_temp[1] = edge_index_temp[1] + 198
        edge_index = torch.zeros(edge_index_temp.size(), dtype=torch.int64)
        nodes = torch.unique_consecutive(
            edge_index_temp.flatten())  # DO NOT SORT THE ELEMENTS, YOU WILL LOSE THE ORDER OF rels_vecs
        nodes = nodes.tolist()
        for i, row in enumerate(edge_index_temp):
            for j, el in enumerate(row):
                index = nodes.index(el)
                edge_index[i][j] = index
        return torch.unique_consecutive(edge_index, dim=1)
        # NB: given that we summed 198 to the object indices, the first line of edge_index
        # will always be zero in case of verb-object relationship

    def __getitem__(self, idx):
        # Extract data from the dictionary
        # TODO: why these deepcopies???? previous logic was messing up - now it shouldn't be needed
        item = copy.deepcopy(self.graphs[idx])
        clip_features = item['clip_feat']
        obj_feats = item['obj_feats']
        triplets = copy.deepcopy(item['triplets'])
        rels = item['rels_vecs']
        verb_idx = item['verb_idx']
        obj_indices = item['obj_indices']

        # Concatenate clip and object features, pad the object features to match clip features
        clip_features = clip_features.unsqueeze(0)
        obj_feats = torch.cat([obj_feats, torch.zeros(obj_feats.size(0), clip_features.size(1) - obj_feats.size(1))], dim=1)
        x = torch.cat([clip_features, obj_feats], dim=0)

        # Create edge index tensor
        edge_index = self.triplets2edge_index(triplets)

        triplets = copy.deepcopy(item['triplets']) # LEAVE THIS HERE, triplets is modified in triplets2edge_index, needs to be reset
        # Create target tensors
        gt_rels = torch.zeros((391,14))
        gt_rels[:,-1]=1
        for el in triplets:
            gt_rels[el[1],el[2]] = 1
            gt_rels[el[1],-1] = 0

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)
        
        # TODO: don't need to encapsulate the GTs in pytorch geometric structure, these have the same size for each elem. in batch!
        return data, verb_idx, gt_rels