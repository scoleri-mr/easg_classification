import torch
import random
from torch.utils.data import Dataset, DataLoader

"""
    File to create the dataset for the anticipation task.
    Requires the saved 'encoded_videos_{split}_{latent_dim}.pth file that
    contains the encoded versions of the frames obtained with a trained vae.
    The getitem returns a random window from the video at the given index
"""
def filter_short_videos(dataset_original, threshold):
    long_videos = []
    for video_id, frames in dataset_original.items():
        if len(frames)>=threshold:
            long_videos.append(frames)
    return long_videos

def random_window_selection(video, window_size):
    num_frames = video.size(0)
    if num_frames == window_size:
        random_start = 0    #if I have exactly the number of frames as the window_size return the whole video, random.randint doesn't work
    else: 
        random_start = random.randint(0, num_frames - window_size - 1)
    return video[random_start:random_start+window_size]

def get_subvideos(self, train_video_original):
    final_videos = []
    for video in train_video_original:
        k = (int((len(video)-self.window)/self.shift))+1   # number of subvideos that can be extracted from the current video
        for i in range(k):
            final_videos.append(video[i*self.shift : i*self.shift+self.window])
    return final_videos

class EASGvideo(Dataset):
    def __init__(self, dataset_path, threshold=20, window_size=20, original=False):
        self.window_size = window_size
        self.threshold = threshold
        self.original = original

        self.dataset_original = torch.load(dataset_path)
        self.long_videos = filter_short_videos(self.dataset_original, self.threshold)

        if self.window_size > self.threshold:
            raise Exception("window_size > threshold, may try to get more frames than available.")
 
    def __len__(self):
        return len(self.long_videos)

    def __getitem__(self, idx):
        return random_window_selection(self.long_videos[idx], self.window_size)
    
def main():
    dataset = EASGvideo("easg_classification/dataset_video/encoded_videos_train_256.pth", threshold=20, window_size=20, original=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Print dataset size
    print(f"Total videos: {len(dataset)}")
    
    # Iterate through the DataLoader and print some samples
    for batch_idx, batch_data in enumerate(dataloader):
        print(f"Batch {batch_idx+1}:")
        print(batch_data)
        if batch_idx >= 2:  # Limit to 3 batches for debugging
            break

if __name__ == "__main__":
    main()