# Libs import
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils

# My imports
from architectures import UNet3D, UNet, Unet2.5D
from dataloader import BddDataset, BddDataloader
from train import train_model, test_model
from loss import LossFunction
from utils import log
from utils.metrics import calc_metrics


def run_finetuning(run_name, results_path='results/', batch_size, max_samples, data_path, exposure, window_size, offset, model_name, model_state):
    # Hiperparameters and configurations
    params = {
        "RUN_NAME": run_name,
        "RESULTS_PATH": results_path,
        "RUN_PATH": results_path+run_name+'/',
        "SEED": 12,
        "BATCH_SIZE": batch_size,
        "MAX_SAMPLES": max_samples,
        "DATA_PATH" data_path,
        "TRAIN_FILE_PATH": data_path + 'bdd_day_train.csv',
        "TEST_FILE_PATH": data_path + 'bdd_day_test.csv'
        "EXPOSURE": data_path,
        "WINDOW_SIZE": window_size,
        "OFFSET": offset,
        "LOG_INTERVAL": 100,  # sample unit
        "TEST_INTERVAL": 1000,  # sample unit
        "CHECKPOINT_INTERVAL": 1000,  # sample unit
        "MODEL_STATE_NAME": model_state,
        "MODEL_STATE_PATH": 'results/{0}/weights/{0}_{1}.pth'.format(model_name, model_state)
    }


    # Set host or device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Create a fodler for results
#     try:
#         os.mkdir(RUN_PATH)
#         os.mkdir(RUN_PATH+'/test_images')
#         os.mkdir(RUN_PATH+'/val_images')
#         os.mkdir(RUN_PATH+'/weights')
#     except:
#         sys.exit("Reset result folder: {}".format(params['RUN_PATH']))

    # Log in file
    sys.stdout = open('{}results3.csv'.format(params['RUN_PATH']), 'w')

    # Set seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Set dataloaders
    train_dataset = BddDataset(params['TRAIN_FILE_PATH'], params['DATA_PATH'], params['EXPOSURE'],
                               params['BATCH_SIZE'], window_size=params['WINDOW_SIZE'], offset=params['OFFSET'])
    train_loader = BddDataloader(train_dataset, params['BATCH_SIZE'], num_workers=4)

    test_dataset = BddDataset(params['TEST_FILE_PATH'], params['DATA_PATH'], [6],
                              params['BATCH_SIZE'], window_size=params['WINDOW_SIZE'], validation=True,
                              offset=params['OFFSET'])
    test_loader = BddDataloader(test_dataset, params['BATCH_SIZE'], num_workers=4, shuffle=False)

    # Set model
    model = UNet3D.UNet3D(3, 3).to(device)
    # model = UNet.UNet(3, 3).to(device)
    #model.load_state_dict(torch.load(MODEL_STATE_PATH))

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Set criterion
    criterion = LossFunction().to(device)

    # Log model configurations
    # log.log_model_eval(model)
    # log.log_model_params(model)

    n_samples = MODEL_STATE_NAME

    print('Batch;TotalLoss;AvgLoss;AvgSsim;AvgPsnr')
    while (n_samples < MAX_SAMPLES):

        train_loss = []
        # Iterate over train loader
        for _, sample in enumerate(train_loader):

            n_samples += 1

            # Send data to device
            x = sample['x'].to(device=device, dtype=torch.float)
            y = sample['y'].to(device=device, dtype=torch.float)

            x = torch.squeeze(x, 2) if WINDOW_SIZE == 1 else x

            # Train model with sample
            _, loss = train_model(model, {'x': x, 'y': y}, criterion, optimizer)

            train_loss.append(float(loss))

            # Log loss
            if n_samples % LOG_INTERVAL == 0:

                print('{};{:.6f};{:.6f};;'
                      .format(n_samples, np.sum(train_loss), np.average(train_loss)))
                train_loss = []

            # Test model
            if n_samples % TEST_INTERVAL == 0:
                test_loss = []
                test_metrics = []

                with torch.no_grad():
                    for test_step, sample in enumerate(test_loader):

                        # Send data to device
                        x = sample['x'].to(device=device, dtype=torch.float)
                        y = sample['y'].to(device=device, dtype=torch.float)

                        x = torch.squeeze(x, 2) if WINDOW_SIZE == 1 else x

                        # Test model with sample
                        outputs, loss = test_model(model, {'x': x, 'y': y}, criterion, optimizer)
                        test_loss.append(float(loss))
                        test_metrics.extend(calc_metrics(outputs, y))

                        # Save first test sample
                        if test_step == 0:
                            log.log_images(x, y, outputs, '{}{}/{}_'
                                            .format(RUN_PATH, 'test_images', n_samples), BATCH_SIZE, WINDOW_SIZE)

                    #break

                # Logs after test
                print('Test;{:.6f};{:.6f};{:.6f};{:.6f}'
                      .format(np.sum(test_loss), np.average(test_loss),
                              np.average([m[0] for m in test_metrics]), np.average([m[1] for m in test_metrics])))
                              #np.average(test_metrics[0]), np.average(test_metrics[1])))

            # Checkpoint
            if n_samples % CHECKPOINT_INTERVAL == 0:
                torch.save(model.state_dict(), '{}{}/{}_{}.pth'
                           .format(RUN_PATH, 'weights', RUN_NAME, n_samples))
