from __future__ import print_function, division
import os
import sys
import pims
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

def rescale(x):
    return x / 255.0

def transforms_list():
    return [
        #transforms.ToPILImage(),
        transforms.Resize((400, 720)),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: rescale(x)),
        transforms.Normalize(mean=(0.279, 0.293, 0.290), std=(0.197, 0.198, 0.201))
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

def custom_collate(batch):
    #print('**')
    #print(len(batch))
    #print(batch[0]['x'].shape)
    #print(batch[0]['y'].shape)
    #for item in batch:
    #    print(item['x'].shape)

    data = torch.stack([item['x'] for item in batch], dim=0)
    target = torch.stack([item['y'] for item in batch], dim=0)
    #print('**')
    #print(data.shape)
    #print(target.shape)

    return {'x':data, 'y':target}


class BddDaloaderFactory():

    def __init__(self, exposure, video_path, batch_size, chunksize=198):

        if exposure == 'under':
            self.gamma = [0.1, 0.2, 0.4]
        elif exposure == 'over':
            self.gamma = [2.5, 5, 10]
        else:
            sys.exit("O tipo de exposiçao deve ser 'under' ou 'over'!")

        self.chunksize = chunksize
        self.video_path = video_path
        self.batch_size = batch_size

    def iterate(self):
        for df in pd.read_csv(self.video_path, sep=',', chunksize=self.chunksize): #ajustar
        
            print('Dataloader iterate:')
            print('\t', end='')
            print(datetime.datetime.now())

            #df x 200
            video_path = df['video_path'].tolist()[0] # str
            target_frames = df['target_frame'].tolist() # list
            #print(target_frames)
                
            window_frames = [[int(i) for i in x.split('-')] for x in df['frames_list']] # list of lists

            dataset = SingleVideoDataset(video_path, target_frames, window_frames, random.choice(self.gamma))

            dataloader = DataLoader(dataset=dataset, 
                                    batch_size=self.batch_size, 
                                    num_workers=0,
                                    collate_fn=custom_collate)

            yield dataloader


class SingleVideoDataset(Dataset):

    def __init__(self, video_path, target_frames, window_frames, gamma, transform=transforms.Compose(transforms_list())):

        self.video_loader = VideoLoader(video_path, (len(window_frames[0])+1))
        #self.video_loader = pims.PyAVReaderIndexed(video_path)
        self.targets = target_frames
        #self.frames = window_frames
        self.transform = transform
        self.gamma = gamma
        #self.shape = (len(self.samples),) + video[0].shape()

    def __len__(self):
        #print('>>')
        #print(self.targets.size)
        return len(self.targets)
    
    def __getitem__(self, idx):

        print('VideoDataset get item:')
        print('\t', end='')
        print(datetime.datetime.now())
        #print(idx)
        #print(self.targets[idx])
        #frame_gt = self.video_loader[self.targets[idx]]
        frames = self.video_loader.process()

        frame_gt = frames[int(len(frames)/2)]
        #frame_gt = ndimage.rotate(frame_gt, 90, reshape=True)

        #frames_concat = [self.transform(self.change_gamma(frame_gt, self.gamma))] #ajustar
        #print(self.frames[idx])
        #print(frame_gt)        

        frame_gt = transforms.functional.to_pil_image(frame_gt)
            
        frame_gt = self.transform(frame_gt)
        #print(frame_gt)
        #print(frame_gt.shape)

        stack = []
        for frame in frames:
            frame = self.change_gamma(frame, self.gamma)
            frame = self.transform(frame)
            stack.append(frame)

        #print(frames)
        #print(type(frames))
        #print(type(frames[0]))

        stack = torch.stack(stack, dim=0)

        #print(stack.shape)
        #print(stack)
        #print(frame_gt.shape)
        #print(frame_gt)

        #print(frames_concat.shape)
        sample = {
            'x': stack,
            'y': frame_gt
        }
        #print(frame_gt)

        return sample

    def change_gamma(self, f, gamma):
        f = transforms.functional.to_pil_image(f)
        f = transforms.functional.adjust_gamma(f, gamma, gain=1)

        return f

class VideoLoader():

    def __init__(self, video_path, window_size):
        self.cap = cv2.VideoCapture(video_path)
        self.window_size = window_size
        self.index = 0
        self.frames = []

    
    def process(self):
        print('VideoLoader process:')
        print('\t', end='')
        print(datetime.datetime.now())
        
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