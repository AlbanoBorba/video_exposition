import os
import csv
import cv2
import pandas as pd
import random
import numpy as np

def split_dataset(csv_file, data_path):
    df = pd.read_csv(csv_file)
    
    split = {
        'bdd_day_train.csv': df.iloc[0:7000, :],
        'bdd_day_val.csv': df.iloc[7000:9000, :],
        'bdd_day_test.csv': df.iloc[9000:-1, :]
    }
    
    for key, value in split.items():
        count = 0
        for index, v in value.iterrows():
            if count == 0:
                write_video_path(data_path+'/'+key, 'video_path')
                count += 1
            write_video_path(data_path+'/'+key, v['video_path'])
        
        
def write_video_path(out_file, video_name):
    with open(out_file, mode='a') as f:
        writer = csv.writer(f, delimiter=',')

        writer.writerow([video_name])
        
if __name__ == '__main__':

    out_path = '/home/albano/Documents/bdd_images/bdd_day_all.csv'
    data_path = '/home/albano/Documents/bdd_images'

    split_dataset(out_path, data_path)
    
#     with open(out_path, mode='w') as outfile:
#         writer = csv.writer(outfile, delimiter=',')
#         writer.writerow(['video_path'])

#     for folder in os.listdir(data_path):
#         write_video_path(out_path, folder)