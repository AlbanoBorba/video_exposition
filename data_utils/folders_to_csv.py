import os
import csv
import cv2
import pandas as pd

def insert_video_samples_in_csv(out_file_name, video_path, video_length, window_length=3, max_samples=20, fixe_window=False):
    
    samples = []

    for i in range(max_samples): #target frame per video
        
        target = i 
        frames = ''
        
        for l in range(1, int(window_length/2)+1): #frame per window size
            if(i - l >= 0): 
                frames = frames + str(i - l) + '-'
            if(i + l <= video_length):
                frames = frames + str(i + l) + '-'     
        if fixe_window == False:
            samples.append([video_path, target, frames[:-1]])
        elif len(frames.split('-'))-1 == (window_length-1):
            #print(len(frames.split('-')))
            samples.append([video_path, target, frames[:-1]])
    
    with open(out_file_name, mode='a') as outfile:
        writer = csv.writer(outfile, delimiter=',')

        for sample in samples:
            writer.writerow(sample)

def read_video(path):
    cap = cv2.VideoCapture(path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return video_length

if __name__ == '__main__':
    
    out_path = 'bdd_night_train_5k.csv'
    in_path = '/media/albano/external'
    op = 'train' #train, val or test
    max_videos = 5 * 1000
    
    with open(out_path, mode='a') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['video_path', 'target_frame', 'frames_list'])

    samples = []
    count = 0
    for root, dirs, files in os.walk(in_path):
        for f in files:
            if op in root:
                if f.endswith('.mov') and f[0] == 'n':
                    video_length = read_video(os.path.join(root,f))
                    insert_video_samples_in_csv(out_path, os.path.join(root,f), video_length, fixe_window=True)
                            
                    if count == max_videos: break
                    else: count = count + 1


    for df in pd.read_csv(out_path, sep=',', chunksize=1):
        #print(df['video_path'].item())
        #print(df['target_frame'].item())
        #print(df['frames_list'].item())
        #count = count + 1
        break
    #print(count)
