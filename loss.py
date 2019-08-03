import torch
import torch.nn as nn
from loss_utils.vgg import Vgg16
from utils import log
from torchvision import transforms
import pytorch_colors as colors

class LossFunction(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        #self.weight = weight
        self.vgg = Vgg16(requires_grad=False)
        self.mse = nn.MSELoss()
        self.mse_vgg = nn.MSELoss()

    def forward(self, x, y):

        # change colorspace
        x = colors.rgb_to_yuv(x)
        y = colors.rgb_to_yuv(y)
        
        # mse loss
        loss_mse = self.mse(x, y)
        
        # feature loss
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss_vgg = self.mse_vgg(x_vgg.relu2_2, y_vgg.relu2_2)
        
        loss = loss_mse + (0.1 * loss_vgg)  # ajustar
        
        return loss
