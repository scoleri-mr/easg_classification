from torch_geometric.data import Data, Dataset as PygDataset
from pathlib import Path
from torch.utils.data import Dataset
import torch
import pickle

"""
    This files allows to create a dataset where the getitem returns a "video":
    starting from the annotated data, groups all the annotated frames by video,
    sorts the frames based on their order, returns a list of pytorch geometric 
    data objects in which each object is an annotated frame.
"""

class EASGvideo_original(PygDataset):
    def __init__(self, path_annts, path_data, split, verbs, objs, rels):
        self.path_annts = path_annts
        self.path_data = path_data
        self.split = split
        self.num_objs = len(objs)
        self.num_verbs = len(verbs)
        self.num_rels = len(rels)
        self.verbs = verbs
        self.objs = objs
        self.rels = rels

        with open(path_annts / f'easg_{split}.pkl', 'rb') as f:
            annts = pickle.load(f)

        with open(path_data / f'roi_feats_{split}.pkl', 'rb') as f:
            roi_feats = pickle.load(f)

        clip_feats = torch.load(path_data / 'verb_features.pt')

        graphs = []
        for graph_uid in annts:
            graph = {}
            for aid in annts[graph_uid]['annotations']:  # cycle on all the annotations of the graph
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

                graph[verb_idx]['frame_id'] = graph_uid
                graphs.append(graph[verb_idx])

        # Group frames by video
        self.videos = {}
        for graph in graphs:
            frame_id = graph['frame_id']
            video_id = frame_id[:36]  # video_id is the first 36 characters
            frame_number = int(graph['frame_id'][37:])
            if video_id not in self.videos:
                self.videos[video_id] = {}
            self.videos[video_id][frame_number] = graph

        self.video_list = list(self.videos.keys())

    def __len__(self):
        return len(self.video_list)

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
        edge_index_temp = triplets.clone()[:, :2].t().contiguous()
        edge_index_temp[1] = edge_index_temp[1] + 198
        edge_index = torch.zeros(edge_index_temp.size(), dtype=torch.int64)
        nodes = torch.unique_consecutive(edge_index_temp.flatten())
        nodes = nodes.tolist()
        for i, row in enumerate(edge_index_temp):
            for j, el in enumerate(row):
                index = nodes.index(el)
                edge_index[i][j] = index
        return torch.unique_consecutive(edge_index, dim=1)

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        frames = self.videos[video_id]
        video_data = []

        for frame_number, graph in frames.items():
            verb_idx = graph['verb_idx']
            clip_features = graph['clip_feat']
            obj_feats = graph['objs']

            obj_indices = torch.zeros(0, dtype=torch.long)
            obj_features = torch.zeros((0, 1024), dtype=torch.float32)
            rels_vecs = torch.zeros((0, len(self.rels)), dtype=torch.float32)
            triplets = torch.zeros((0, 3), dtype=torch.long)

            for obj_idx in graph['objs']:
                obj_indices = torch.cat((obj_indices, torch.tensor([obj_idx], dtype=torch.long)), dim=0)
                obj_features = torch.cat((obj_features, graph['objs'][obj_idx]['obj_feat'].unsqueeze(0)), dim=0)
                rels_vec = graph['objs'][obj_idx]['rels_vec']
                rels_vecs = torch.cat((rels_vecs, rels_vec.unsqueeze(0)), dim=0)

                frame_triplets = []
                for rel_idx in torch.where(rels_vec)[0]:
                    frame_triplets.append((verb_idx, obj_idx, rel_idx.item()))
                triplets = torch.cat((triplets, torch.tensor(frame_triplets, dtype=torch.long)), dim=0)

            clip_features = clip_features.unsqueeze(0)
            obj_features = torch.cat([obj_features, torch.zeros(obj_features.size(0), clip_features.size(1) - obj_features.size(1))], dim=1)
            x = torch.cat([clip_features, obj_features], dim=0)

            edge_index = self.triplets2edge_index(triplets)
            data = Data(x=x, edge_index=edge_index)
            data.verb_idx = torch.tensor([verb_idx], dtype=torch.long)
            data.obj_indices = obj_indices
            data.rels_vecs = rels_vecs
            data.triplets = triplets
            data.video_id = video_id
            data.frame_number = frame_number
            video_data.append(data)

        video_data.sort(key=lambda data: data.frame_number)
        return video_data  # Return a list of Data objects for this video

class EASGvideo_pyg(Dataset):
    def __init__(self, path_annts, path_data, split, verbs, objs, rels, treshold=20, window=20, shift=20, original=False):
        self.window = window
        self.treshold = treshold
        self.shift = shift
        self.original = original

        self.dataset_original = EASGvideo_original(path_annts, path_data, split, verbs, objs, rels)
        short_videos = self.filter_short_videos(self.dataset_original)
        self.final_videos = self.get_subvideos(short_videos)

    def filter_short_videos(self, train_video_original):
        long_videos = []
        for video in train_video_original:
            if len(video)>=self.treshold:
                long_videos.append(video)
        return long_videos
    
    def get_subvideos(self, train_video_original):
        final_videos = []
        for video in train_video_original:
            k = (int((len(video)-self.window)/self.shift))+1   # number of subvideos that can be extracted from the current video
            for i in range(k):
                final_videos.append(video[i*self.shift : i*self.shift+self.window])
        return final_videos
    
    def __len__(self):
        return len(self.final_videos)

    def __getitem__(self, idx):
        if self.original:
            return self.dataset_original[idx]
        else: return self.final_videos[idx]