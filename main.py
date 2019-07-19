import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, utils
from skimage import io

from models import UNet3D
from dataloader import BddDaloaderFactory
from train import train_model
from loss import LossFunction

SEED = 12
BATCH_SIZE = 8
TRAIN_FILE_PATH = 'data_utils/bdd_day[90-110]_train_5k_40.csv'
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
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#print(params)

train_model(model, train_loader, criterion, optimizer)