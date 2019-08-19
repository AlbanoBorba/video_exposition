# Libs import
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils

# My imports
from architectures import UNet3D
from dataloader import BddDataset, BddDataloader
from train import train_model, test_model
from loss import LossFunction
from utils import log

# Hiperparameters and configurations
RUN_NAME = 'max_loss_v3'
RESULTS_PATH = 'results/'
RUN_PATH = RESULTS_PATH+RUN_NAME+'/'
SEED = 12
BATCH_SIZE = 4
EPOCHS = 10
DATA_PATH = '~/Documents/bdd_images/'
TRAIN_FILE_PATH = DATA_PATH + 'bdd_day_train.csv'
TEST_FILE_PATH = DATA_PATH + 'bdd_day_test.csv'
EXPOSURE = 'under'
WINDOW_SIZE = 3
LOG_INTERVAL = 100  # sample unit
TEST_INTERVAL = 1000  # sample unit
CHECKPOINT_INTERVAL = 2000  # sample unit

# Set host or device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()

# Create a fodler for results
try:
    os.mkdir(RUN_PATH)
    os.mkdir(RUN_PATH+'/test_images')
    os.mkdir(RUN_PATH+'/val_images')
    os.mkdir(RUN_PATH+'/weights')
except:
    sys.exit("Reset result folder: {}".format(RUN_PATH))

# Log in file
sys.stdout = open('{}results.csv'.format(RUN_PATH), 'w')

# Set seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Set dataloaders
train_dataset = BddDataset(TRAIN_FILE_PATH, DATA_PATH, EXPOSURE,
                           BATCH_SIZE, window_size=WINDOW_SIZE)
train_loader = BddDataloader(train_dataset, BATCH_SIZE, num_workers=4)

test_dataset = BddDataset(TEST_FILE_PATH, DATA_PATH, EXPOSURE,
                          BATCH_SIZE, window_size=WINDOW_SIZE)
test_loader = BddDataloader(test_dataset, BATCH_SIZE, num_workers=4)

# Set model
model = UNet3D.UNet3D(3, 3).to(device)

# Set optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters())

# Set criterion
criterion = LossFunction().to(device)

# Log model configurations
# log.log_model_eval(model)
# log.log_model_params(model)

n_samples = 0

print('Video;TotalLoss;AvgLoss')
for epoch in range(EPOCHS):
    #log.log_time('Epoch {}/{}'.format(epoch, EPOCHS - 1))

    train_loss = []
    # Iterate over train loader
    for _, sample in enumerate(train_loader):

        n_samples += 1

        # Send data to device
        x = sample['x'].to(device=device, dtype=torch.float)
        y = sample['y'].to(device=device, dtype=torch.float)

        # Train model with sample
        _, loss = train_model(model, {'x': x, 'y': y}, criterion, optimizer)

        train_loss.append(float(loss))

        # Log loss
        if n_samples % LOG_INTERVAL == 0:

            print('{};{:.6f};{:.6f}'
                  .format(n_samples, np.sum(train_loss), np.average(train_loss)))
            train_loss = []

        # Test model
        if n_samples % TEST_INTERVAL == 0:
            test_loss = []

            for test_step, sample in enumerate(test_loader):

                # Send data to device
                x = sample['x'].to(device=device, dtype=torch.float)
                y = sample['y'].to(device=device, dtype=torch.float)

                # Test model with sample
                outputs, loss = test_model(model, {'x': x, 'y': y}, criterion)
                test_loss.append(float(loss))

                # Save first test sample
                if test_step == 0:
                    log.log_images(x, y, outputs, '{}{}/{}_'
                                    .format(RUN_PATH, 'test_images', n_samples), BATCH_SIZE)

            # Logs after test
            print('Test;{:.6f};{:.6f}'
                  .format(np.sum(test_loss), np.average(test_loss)))

        # Checkpoint
        if n_samples % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), '{}{}/{}_{}.pth'
                       .format(RUN_PATH, 'weights', RUN_NAME, n_samples))
