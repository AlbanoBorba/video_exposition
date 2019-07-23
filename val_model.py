# Libs import
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils

# My imports
from models import UNet3D
from dataloader import BddDaloaderFactory
from train import train_model, test_model
from loss import LossFunction
from utils import log

# Hiperparameters and configurations
RUN_NAME = ''
BATCH_SIZE = 4
VAL_FILE_PATH = './data_utils/csv_loaders/bdd_day[90-110]_train_5k_40.csv'
MODEL_STATE_PATH = './results/3dcnn_weigths_4.pth'
EXPOSURE = 'under'

# Set host or device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set dataloaders
val_loader = BddDaloaderFactory(VAL_FILE_PATH, EXPOSURE, BATCH_SIZE, n_videos=5)

# Set model and lod weights
model = UNet3D(3, 3).to(device)
model.load_state_dict(torch.load(MODEL_STATE_PATH))

# Set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters())

# Set criterion
criterion = LossFunction().to(device)

val_loss = []
# Iterate over videos.
for video_step, video_loader in val_loader.iterate():
    # Iterate over frames.
    for sample_step, sample in enumerate(video_loader):

        # Send data to device
        y, x = sample['y'].to(device), sample['x'].to(device)

        # Test model with sample
        outputs, loss = test_model(model, {'x': x, 'y': y}, criterion, optimizer)
        val_loss.append(loss)
        print(loss)

        if sample_step == 0:
            log.log_images(x, y, outputs,'./results/')

# Logs after test
#log.log_time('Total Loss: {:.6f}\tAvg Loss: {:.6f}'.format(np.sum(val_loss), np.average(val_loss)))
