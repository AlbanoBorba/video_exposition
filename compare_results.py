import os
import csv
import cv2
import pandas as pd
import numpy as np
import random

files = [
    #('mix_loss', 'results/mix_loss_v3_adam/val_images/eval_300000_under_4.csv'),
    ('mix_plus_vgg', 'results/mix_plus_vgg/val_images/eval_300000_under_4.csv'),
]

print('Name;ssim_avg;ssim_std;psnr_avg;psnr_std')
for name, file in files:
    df = pd.read_csv(file, sep=';')

    ssim_avg = np.average(df['ssim'])
    ssim_std = np.std(df['ssim'])

    psnr_avg = np.average(df['psnr'])
    psnr_std = np.std(df['psnr'])

    print('{};{};{};{};{}'.format(name, ssim_avg, ssim_std, psnr_avg, psnr_std))