import torch
import torch.nn as nn
from loss_utils.vgg import Vgg16

class LossFunction(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        #self.weight = weight
        self.vgg = Vgg16(requires_grad=False)
        self.mse = nn.MSELoss()
        

    def forward(self, x, y):
        
        #print('*'*10)
        #print(x.shape)
        #print(y.shape)
        loss_mse = self.mse(x, y)
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        #print(x_vgg.relu2_2.shape)
        #print(y_vgg.relu2_2.shape)
        loss_vgg = self.mse(x_vgg.relu2_2, y_vgg.relu2_2)

        loss = loss_mse + loss_vgg #ajustar

        return loss