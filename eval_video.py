import os
import sys
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import skimage.exposure
import torch
from torchvision import transforms, utils
from architectures import UNet3D, UNet
from global_motion import sparse_optical_flow

def transforms_list():
    return [
        transforms.ToPILImage(),
        transforms.Resize((400, 720)),
        transforms.CenterCrop((400, 400)),
        #transforms.Lambda(lambda x: ndimage.rotate(x, 90, reshape=True)),
        #transforms.ToTensor(),
    ]

def change_gamma(f, gamma):
    # f = transforms.functional.to_pil_image(f)
    f = transforms.functional.adjust_gamma(f, gamma)
    # f = skimage.exposure.adjust_gamma(f, gamma)

    return f

def read_video(video_path, data_path, window_size, n_frames, dilatation=1):
    
    transform = transforms.Compose(transforms_list())
    windows = []
    targets = []
    offset = int(window_size/2) + int(dilatation/2)

    for target in range(offset, n_frames-offset+1):
        window = []
        for i in range(target-offset, target+offset+1, dilatation):
            window.append(i)
        windows.append(window)
        targets.append(target)
    # print(len(windows))

    video = []
    for t, w in zip(targets, windows):
        video.append({
            'target': ['{}/{:02d}.png'.format(video_path, t)], 
            'window': ['{}/{:02d}.png'.format(video_path, x) for x in w]
            })

    # print(video)

    for v in video: 
        v['target'] = [transform(frame) for frame in io.imread_collection([data_path + x for x in v['target']])]
        v['window'] = [transform(frame) for frame in io.imread_collection([data_path + x for x in v['window']])]
    
    return video

def restore(frames, window_size, dilatation, gamma, degradation='under'):
    
    MODEL_STATE_PATH = './../tcc_data/' + '3_under.pth'
    device = 'cpu'
    
    model = UNet3D.UNet3D(window_size).to(device)
    model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=torch.device('cpu')))

    outputs = []
    for f in frames:
        f = torch.stack([transforms.functional.to_tensor(x) for x in f], dim=1).unsqueeze(0)
        # print('shape: ', f.shape)
        outputs.append(model(f))

    return outputs

def eval_videos(videos, data_path, window_size, n_frames, dilatation, gamma=6, metrics=[]):
    
    results = []
    for path in videos.iterrows():
        path = path[1].values[0]

        video = read_video(path, data_path, window_size, n_frames, dilatation)

        original = [v['target'][0] for v in video]
        exposed = [change_gamma(video, gamma) for video in original]
        restored = restore([v['window'] for v in video], window_size, dilatation, gamma)

        # CALC_METRICS
        results.append([
            sparse_optical_flow(original),
            sparse_optical_flow(exposed),
            sparse_optical_flow(restored)
            ])
        
    return results

if __name__=="__main__":

    data_path = '../tcc_data/'
    # csv_path = data_path + 'bdd_day_val.csv'
    csv_path = data_path + 'teste.csv'
    max_videos = 10
    
    results = eval_videos(pd.read_csv(csv_path, nrows=max_videos), data_path, 3, 50, 1)
    print(results)