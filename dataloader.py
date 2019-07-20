from __future__ import print_function, division
import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import cv2
import time
import datetime
from scipy import ndimage, misc
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def transforms_list():
    return [
        #transforms.ToPILImage(),
        transforms.Resize((400, 720)),
        transforms.CenterCrop((400, 400)),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: rescale(x)),
        transforms.Normalize(mean=(0.279, 0.293, 0.290), std=(0.197, 0.198, 0.201))
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

def custom_collate(batch):
    data = torch.stack([item['x'] for item in batch], dim=0)
    target = torch.stack([item['y'] for item in batch], dim=0)

    return {'x':data, 'y':target}


class BddDaloaderFactory():

    def __init__(self, csv_path, exposure, batch_size, n_samples=40, window_size=3):

        if exposure == 'under':
            self.gamma = [0.1, 0.2, 0.4]
        elif exposure == 'over':
            self.gamma = [2.5, 5, 10]
        else:
            sys.exit("O tipo de exposiçao deve ser 'under' ou 'over'!")

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.windo_size = window_size
        self.video_loader = pd.read_csv(csv_path)

    def __len__(self):
        return len(video_loader.index)

    def __getitem__(self):
        random_video = video_loader.sample(n=1)
        video_path = random_video['video_path'].tolist()[0] # str

        dataset = SingleVideoDataset(video_path, self.n_samples, self.window_size, random.choice(self.gamma))

        dataloader = DataLoader(dataset=dataset, 
                                batch_size=self.batch_size, 
                                num_workers=0,
                                collate_fn=custom_collate)

        return dataloader


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

        # Preprocess window
        stack = []
        for frame in frames:
            frame = self.change_gamma(frame, self.gamma)
            frame = self.transform(frame)
            stack.append(frame)

        stack = torch.stack(stack, dim=0)

        # Set sampleW
        sample = {
            'x': window,
            'y': frame_gt
        }

        return sample

    def change_gamma(self, f, gamma):
        f = transforms.functional.to_pil_image(f)
        f = transforms.functional.adjust_gamma(f, gamma, gain=1)

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
            self.frames.append(frame)

        return self.frames
            

if __name__ == '__main__':
    
    file_path = 'data_utils/bdd_night_train.csv'

    for epoch in range(epochs):
        for videoDataloader in bddDaloaderFactory('under', file_path, 8):
            for step, sample in enumerate(videoDataloader):
                y, x = sample["y"], sample["x"]
                print(step, y.shape, x.shape)