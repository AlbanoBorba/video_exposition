import torch
import torch.nn as nn
from loss_utils.vgg import Vgg16
from utils import log
from torchvision import transforms


def transform_list():
    return [
        transforms.ToPILImage(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ]

class LossFunction(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        #self.weight = weight
        self.vgg = Vgg16(requires_grad=False)
        self.mse = nn.MSELoss()
        self.mse_vgg = nn.MSELoss()
        self.transform = transforms.Compose(transform_list())


    def forward(self, x, y):

        # print('*'*10)
        # print(x.shape)
        # print(y.shape)

        loss_mse = self.mse(x, y)
        x_vgg = self.vgg(x)#self.transform(x))
        y_vgg = self.vgg(y)#self.transform(y))

        #log.log_images_vgg(x_vgg.relu2_2, y_vgg.relu2_2, './results/')

        #print('Shape vgg:')
        #print(x_vgg.relu2_2.shape)
        #print(y_vgg.relu2_2.shape)
        loss_vgg = self.mse_vgg(x_vgg.relu2_2, y_vgg.relu2_2)
        #print('\nLoss vgg: ', loss_vgg)

        loss = loss_mse + (0.1 * loss_vgg)  # ajustar
        #print('\nLoss total: ', loss)
        #print('\n')

        return loss
