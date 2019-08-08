import torch
import torch.nn as nn
from loss_utils.vgg import Vgg16
from loss_utils.models import PerceptualLoss
from utils import log
from torchvision import transforms
import numpy as np
#import pytorch_colors as colors

class LossFunction(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        #self.weight = weight
        #self.vgg = Vgg16(requires_grad=False)
        #self.mse = nn.MSELoss()
        self.perceptual = PerceptualLoss(model='net', net='vgg')
        #self.mse_vgg = nn.MSELoss()

    def to_yuv(self, in_tensor):
        out_tensor = torch.zeros(in_tensor.shape)

        for i, o in zip(in_tensor, out_tensor):
            o[0] =  0.299*i[0] + 0.587*i[1] + 0.114*i[2]
            o[1] = -0.147*i[0] + 0.289*i[1] + 0.436*i[2]
            o[2] =  0.615*i[0] + 0.515*i[1] + 0.100*i[2]

        in_tensor = out_tensor.to(cuda)
        return in_tensor 
        
    def forward(self, x, y):

        # change colorspace
        #x = colors.rgb_to_yuv(x)
        #y = colors.rgb_to_yuv(y)
        
        # mse loss
        #loss_mse = self.mse(self.to_yuv(x), self.to_yuv(y))
        
        # feature loss
        #x_vgg = self.vgg(x)
        #y_vgg = self.vgg(y)
        #loss_vgg = self.mse_vgg(x_vgg.relu2_2, y_vgg.relu2_2)
        
        #loss = loss_mse# + (0.3 * loss_vgg)  # ajustar

        
        loss = self.perceptual(y, x)

        
        return loss
