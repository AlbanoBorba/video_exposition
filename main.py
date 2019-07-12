import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils
from skimage import io

from models import UNet3D
from dataloader import BddDaloaderFactory
from train import train_model
from loss import LossFunction

SEED = 6
BATCH_SIZE = 8
TRAIN_FILE_PATH = 'data_utils/bdd_night_train_5k_40.csv'
EXPOSURE = 'under'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_loader = BddDaloaderFactory(EXPOSURE, TRAIN_FILE_PATH, BATCH_SIZE)

model = UNet3D(3, 3).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters())

criterion = LossFunction().to(device)

#print(model.eval())

train_model(model, train_loader, criterion, optimizer)

'''
img0 = io.imread('test/test_1.jpg')
img1 = io.imread('test/test_2.jpg')
img2 = io.imread('test/test_3.jpg')

x = np.stack((img0, img1, img2), axis=0) #SEMPRE CONCATENAR O 'TARGET FRAME PRIMEIRO'
x = np.moveaxis(x, -1, 0)
x = x.reshape(1, 3, 3, 512, -1)

x = torch.from_numpy(x).to(device=device, dtype=torch.float)

y = model(x)

print(y.shape)
'''