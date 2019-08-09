import os
import numpy as np
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt
import csv
import pandas as pd


def read_video(path, n_frames=50, out_path):
    cap = cv2.VideoCapture(path)
    count = 0

    # cria um novo folder para o vídeo

    while True:
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # preprocessa o frame
            # salva o frame

            count += 1
            if count == n_frames:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()

if __name__ == '__main__':
    csv_path = './csv_loaders/bdd_day[90-110]_all.csv'
    out_path = ''
    
    csv_file = pd.read_csv(csv_path)

    for index, row in csv_file.iterrows():
        read_video(row.values[0])

        # salvo os frames em pasta por vídeo no hd
