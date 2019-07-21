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
BATCH_SIZE = 8
VAL_FILE_PATH = ''
MODEL_STATE_PATH = ''
EXPOSURE = 'under'

# Set dataloaders
val_loader = BddDaloaderFactory(EXPOSURE, TRAIN_FILE_PATH, BATCH_SIZE)

# Set model and lod weights
model = UNet3D(3, 3).to(device)
model.load_state_dict(torch.load(MODEL))

# Set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters())

# Set criterion
criterion = LossFunction().to(device)

val_loss = []
# Iterate over videos.
for video_step, video_loader in enumerate(val_loader):
    # Iterate over frames.
    for _, sample in enumerate(video_loader):

        # Send data to device
        y, x = sample['y'].to(device), sample['x'].to(device)

        # Test model with sample
        loss = test_model(model, {'x': x, 'y': y}, criterion, optimizer)
        test_loss.append(loss)
        #log.log_images(x, y,'<PATH>/{}_'.format(n_samples))

# Logs after test
log.log_time('Test: {}\tTotal Loss: {:.6f}\tAvg Loss: {:.6f}'
             .format(n_samples, np.sum(test_loss), np.average(test_loss)))
