import torch
import torch.nn as nn
from loss_utils.vgg import Vgg16
from utils import log
from torchvision import transforms
#import pytorch_colors as colors

class LossFunction(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        #self.weight = weight
        #self.vgg = Vgg16(requires_grad=False)
        self.mse = nn.MSELoss()
        #self.mse_vgg = nn.MSELoss()

    def to_yuv(self, x):
        out = np.zeros(x.shape)

        out[0] =  0.299*x[0] + 0.587*x[1] + 0.114*x[3]
        out[1] = -0.147*x[0] + 0.289*x[1] + 0.436*x[3]
        out[2] =  0.615*x[0] + 0.515*x[1] + 0.100*x[3]

        return out
        
    def forward(self, x, y):

        # change colorspace
        #x = colors.rgb_to_yuv(x)
        #y = colors.rgb_to_yuv(y)
        
        # mse loss
        loss_mse = self.mse(to_yuv(x), to_yuv(y))
        
        # feature loss
        #x_vgg = self.vgg(x)
        #y_vgg = self.vgg(y)
        #loss_vgg = self.mse_vgg(x_vgg.relu2_2, y_vgg.relu2_2)
        
        loss = loss_mse# + (0.3 * loss_vgg)  # ajustar
        
        return loss
