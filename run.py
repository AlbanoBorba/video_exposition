# Libs import
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils
import os

# My imports
from models import UNet3D
from dataloader import BddDaloaderFactory
from train import train_model, test_model
from loss import LossFunction
from utils import log

# Hiperparameters and configurations
RUN_NAME = 'experiment_refactory_load_image'
RESULTS_PATH = 'results/'
SEED = 12
BATCH_SIZE = 8
EPOCHS = 10
TRAIN_FILE_PATH = 'data_utils/csv_loaders/bdd_day[90-110]_train_5k_40.csv'
TEST_FILE_PATH = 'data_utils/csv_loaders/bdd_day[90-110]_test_5k_40.csv'
EXPOSURE = 'under'
TEST_INTERVAL = 500  # video unit
CHECKPOINT_INTERVAL = 500  # video unit

# Set host or device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()

# Create a fodler for results
os.mkdir(RESULTS_PATH+RUN_NAME)

# Set seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Set dataloaders
train_loader = BddDaloaderFactory(TRAIN_FILE_PATH, EXPOSURE, BATCH_SIZE)
test_loader = BddDaloaderFactory(TEST_FILE_PATH, EXPOSURE, BATCH_SIZE, n_videos=1, n_samples=1)

# Set model
model = UNet3D(3, 3).to(device)

# Set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters())

# Set criterion
criterion = LossFunction().to(device)

# Log model configurations
# log.log_model_eval(model)
log.log_model_params(model)

n_samples = 0
for epoch in range(EPOCHS):
    log.log_time('Epoch {}/{}'.format(epoch, EPOCHS - 1))

    # Iterate over videos.
    for video_step, video_loader in train_loader.iterate():
        video_loss = []

        # Iterate over frames.
        for _, sample in enumerate(video_loader):
            n_samples += 1

            # Send data to device
            y, x = sample['y'].to(device), sample['x'].to(device)

            # Train model with sample
            _, loss = train_model(
                model, {'x': x, 'y': y}, criterion, optimizer)
            #print(loss)
            video_loss.append(float(loss))

        # Logs per video
        log.log_time('Video: {}Total Loss: {:.6f}Avg Loss: {:.6f}'
                     .format(n_samples, np.sum(video_loss), np.average(video_loss)))

        # Test model
        # NOTE: len(train_loader) must be >> len(test_loader)

        if video_step % TEST_INTERVAL == 0:
            test_loss = []

            # Device clear
            #torch.cuda.empty_cache()

            # Iterate over videos.
            for video_step, video_loader in test_loader.iterate():
                    # Iterate over frames.
                for _, sample in enumerate(video_loader):

                    # Send data to device
                    y, x = sample['y'].to(device), sample['x'].to(device)

                    # Test model with sample
                    outputs, loss = test_model(
                        model, {'x': x, 'y': y}, criterion)
                    test_loss.append(float(loss))
                    log.log_images(x, y, outputs, '{}{}/{}_'
                                   .format(RESULTS_PATH, RUN_NAME, n_samples))

            # Logs after test
            log.log_time('Test: {}Total Loss: {:.6f}Avg Loss: {:.6f}'
                         .format(n_samples, np.sum(test_loss), np.average(test_loss)))

        # Checkpoint
        if video_step % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), '{}{}/3dcnn_weigths_{}_{}.pth'
                       .format(RESULTS_PATH, RUN_NAME, epoch, video_step))
