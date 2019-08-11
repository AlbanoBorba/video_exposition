from __future__ import print_function, division
import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import cv2
from skimage import color
from scipy import ndimage, misc
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imageio

def BddDataloader(dataset, batch_size, num_workers):
    
    return dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                num_workers=num_workers,
                                collate_fn=custom_collate)

def custom_collate(batch):
    data = torch.stack([item['x'] for item in batch], dim=0)
    target = torch.stack([item['y'] for item in batch], dim=0)

    return {'x': data, 'y': target}


class BddDaloaderFactory():

    """
    Attributes
    ----------
    csv_path : str (required)
        csv with video url's
    exposure : ('under' or 'over') (required)
        exposition type
    batch_size : int (required)
        batch_size
    window_size : int
        size of the temporal window
    causality : bool
        target frame in the middle of the temporal window if False, else at the end
    offset: int
        offset in the temporal window
    sparsity : bool
        if true, progressive increase offset 
    """

    def __init__(self, csv_path, exposure, batch_size, window_size=3, causality=False, offset=0, sparsity=False):

        if exposure == 'under':
            self.gamma = [2, 4, 6] # [4, 6, 8]
        elif exposure == 'over':
            self.gamma = [0.1, 0.2, 0.4] # [1/4, 1/6, 1/8]
        else:
            sys.exit("Exposition type must be 'under' ou 'over'!")

        self.batch_size = batch_size
        self.window_size = window_size
        self.video_url_loader = pd.read_csv(csv_path)
        self.n_videos = len(video_url_loader.index)

    def __len__(self):
        return self.n_videos

    def __getitem__(self, idx):

        video_url = self.video_url_loader.iloc[idx, :]

        
    def get_sample(transform=transforms.Compose(transforms_list())):
        

    def transforms_list(self):
        return [
            transforms.Resize((400, 720)),
            transforms.CenterCrop((400, 400)),
            transforms.Lambda(lambda x: ndimage.rotate(x, 90, reshape=True)),
            transforms.ToTensor(),
        ]


class SingleVideoDataset(Dataset):

    def __init__(self, video_path, n_samples, window_size, gamma, transform=transforms.Compose(transforms_list())):

        self.sample_loader = SampleLoader(video_path, window_size)
        self.n_samples = n_samples
        self.gamma = gamma
        self.transform = transform

    def __len__(self):
        return self.n_samples

    # Enumerate call
    def __getitem__(self, idx):

        # Get window_size frames
        frames = self.sample_loader.get_sample()

        # Preprocess ground-truth
        frame_gt = frames[int(len(frames)/2)]
        #frame_gt = ndimage.rotate(frame_gt, 90, reshape=True)
        frame_gt = transforms.functional.to_pil_image(frame_gt)
        frame_gt = self.transform(frame_gt)
        #utils.save_image(frame_gt, './results/teste/gt_{}.png'.format(idx))

        # Preprocess window
        window = []
        #count = 0
        for frame in frames:
            frame = self.change_gamma(frame, self.gamma)
            frame = self.transform(frame)
            #utils.save_image(frame, './results/teste/item_{}_{}.png'.format(idx, count))
            #count += 1

            window.append(frame)

        window = torch.stack(window, dim=1)

        # Set sampleW
        sample = {
            'x': window,
            'y': frame_gt
        }

        return sample

    def change_gamma(self, f, gamma):
        f = transforms.functional.to_pil_image(f)
        f = transforms.functional.adjust_gamma(f, gamma)

        return f


class SampleLoader():

    def __init__(self, video_path, window_size):
        self.cap = cv2.VideoCapture(video_path)
        self.window_size = window_size
        self.index = 0
        self.frames = []

    def get_sample(self):

        if self.index == 0:
            for i in range(self.window_size):
                _, frame = self.cap.read()
                self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.index = 1
        else:
            self.frames.pop(0)
            _, frame = self.cap.read()
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return self.frames
