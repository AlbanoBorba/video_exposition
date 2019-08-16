import torch
import torch.nn as nn
from loss_utils.vgg import Vgg16
from utils import log
from torchvision import transforms
import numpy as np
from loss_utils import pytorch_msssim as torch_msssim

def loss_mix_v3(y_true, y_pred):
    
    # weights
    alpha = 0.2
    l1_w = 1-alpha
    msssim_w = alpha
    
    #l1 = K.mean(K.abs(y_pred - y_true)*K.abs(y_true - .5), axis=-1)
    l1_value = torch.mean(torch.abs(y_pred - y_true) * torch.abs(y_true - 0.5))
    #ms_ssim = tf.reduce_mean(1-tf.image.ssim_multiscale(y_pred, y_true, max_val = 1.0))
    msssim_value = torch.mean(1-torch_msssim.msssim(y_pred, y_true)) # must be (0,1) rangee
    
    return (msssim_w*msssim_value) + (l1_w*l1_value)

class LossFunction(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        #self.weight = weight
        #self.vgg = Vgg16(requires_grad=False)
        #self.mse_vgg = nn.MSELoss()
        self.mse = nn.MSELoss()
        
    def forward(self, x, y):
        
        # mse loss
        #loss_mse = self.mse(self.to_yuv(x), self.to_yuv(y))
        
        # mix loss v3
        loss = loss_mix_v3(y, x)

        return loss
