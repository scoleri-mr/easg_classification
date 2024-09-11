"""
    script to create and save the datasets with encoded graphs from the video frames
    Requires trained vae path
    Uses dataset_video.py to get the pytorch geometric frames from the videos
    and saves the encoded frames in a dictionary in which the keys are the videos ids
"""
import argparse
from easg_classification.utils import load_model
from easg_classification.dataset_video.dataset_video_pyg import EASGvideo_original
from pathlib import Path
from torch_geometric.loader import DataLoader
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='CreateEncodedDatasets')
    parser.add_argument('--ann_path', type=str, default='easg_classification/annts_in_new_format/', help='path to annotations')
    parser.add_argument('--data_path', type=str, default='easg_classification/data/', help='path to ROI and clip features')
    parser.add_argument('--vae_path', type=str, help='path to the trained vae', default=None)
    parser.add_argument('--latent_dim', type=int, help='latent dimension to use in the vae', default=256)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the video dataset
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

    train_video = EASGvideo_original(path_annts, path_data, 'train', verbs, objs, rels)
    val_video = EASGvideo_original(path_annts, path_data, 'val', verbs, objs, rels)

    batch_size = 1
    train_loader_v = DataLoader(train_video, batch_size=batch_size, shuffle=False)
    val_loader_v = DataLoader(val_video, batch_size=batch_size, shuffle=False)

    # load the variational autoencoder
    if args.latent_dim == 512:
        vae_path = 'easg_classification/experiments/best_VAE1000_sep=True_fromae=False_od=512_kld=original_b=0.0005_lr=0.0001_fl=True_ex=False_eps=0.1_1722873332/checkpoints/last.ckpt'
    elif args.latent_dim == 256:
        vae_path = 'easg_classification/experiments/best_VAE1000_sep=True_od=256_kld=original_b=0.0005_lr=0.0001_fl=True_ex=False_eps=0.1_1719244127/checkpoints/last.ckpt'
    
    if args.vae_path is not None:
        print("Extracting from custom vae path...")
        vae_path = args.vae_path
        ind = vae_path.find('/checkpoints/last.ckpt')
        run_id = vae_path[ind-4:ind]

    vae = load_model('vae', vae_path, separate=True, output_dim=args.latent_dim)
    vae.eval()

    # from the video datasets, create two dictionaries 
    # (one for train and one for val) with entries:
        # video_id : torch.Tensor of size [num_frames, latent_dim]
    
    encoded_videos_train = {}
    train_triplets = {}
    for video in train_loader_v:
        for frame in video:
            frame.to(device)
            video_id = frame.video_id[0]    # only one video if batch_size=1  
            if video_id not in encoded_videos_train:
                encoded_videos_train[video_id] = []
                train_triplets[video_id] = []
            code = vae.encode(frame).squeeze()
            encoded_videos_train[video_id].append(code)
            train_triplets[video_id].append(frame.triplets)

    for video_id in encoded_videos_train:
        encoded_videos_train[video_id] = torch.stack(encoded_videos_train[video_id])
    
    #save encoded videos and triplets from train dataset
    torch.save(encoded_videos_train, f'easg_classification/dataset_video/encoded_videos_train_{run_id}.pth')
    print("Training video dataset saved!")
    torch.save(train_triplets, f'easg_classification/dataset_video/train_triplets_{run_id}.pth')
    print("Training triplets saved!")

    encoded_videos_val = {}
    val_triplets = {}
    for video in val_loader_v:
        for frame in video:
            frame.to(device)
            video_id = frame.video_id[0]    # only one video if batch_size=1  
            if video_id not in encoded_videos_val:
                encoded_videos_val[video_id] = []
                val_triplets[video_id] = []
            code = vae.encode(frame).squeeze()
            encoded_videos_val[video_id].append(code)
            val_triplets[video_id].append(frame.triplets)

    for video_id in encoded_videos_val:
        encoded_videos_val[video_id] = torch.stack(encoded_videos_val[video_id])

    # save encoded videos and triplets from validation dataset
    torch.save(encoded_videos_val, f'easg_classification/dataset_video/encoded_videos_validation_{run_id}.pth')
    print("Validation video dataset saved!")
    torch.save(val_triplets, f'easg_classification/dataset_video/val_triplets_{run_id}.pth')
    print("Validation triplets saved!")

if __name__ == "__main__":
    main()