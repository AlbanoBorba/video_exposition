import os
import numpy as np
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt
import csv
import pandas as pd


def read_video(path, n_frames=50):
    cap = cv2.VideoCapture(path)
    count = 0
    video_avg = []

    while True:
        ret, frame = cap.read()

        if ret:
            frame_avg = np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            video_avg.append(frame_avg)

            count += 1
            if count == n_frames:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video_length = 100  # int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return video_length, np.average(video_avg)


def read_csv():
    pass


if __name__ == '__main__':
    csv_path = './csv_loaders/bdd_day[90-110]_all.csv'
    out_path = ''
    csv_file = pd.read_csv(csv_path)

    for index, row in csv_file.iterrows():
        print(index, row)

		# abro um vídeo e pego n frames
		# preprocessa o frame
		# salvo os frames em pasta por vídeo no hd

        
