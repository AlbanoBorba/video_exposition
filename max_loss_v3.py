import torch
import pytorch_msssim as torch_msssim

def loss_mix_v3(y_true, y_pred):
    
    # weights
    alpha = 0.2
    l1_w = 1-alpha
    msssim_w = alpha
    
    #l1 = K.mean(K.abs(y_pred - y_true)*K.abs(y_true - .5), axis=-1)
    l1_value = torch.mean(torch.abs(y_pred - y_true) * torch.abs(y_true - 0.5), dim=0)
    #ms_ssim = tf.reduce_mean(1-tf.image.ssim_multiscale(y_pred, y_true, max_val = 1.0))
    msssim_value = torch.mean(1-torch_msssim.msssim(y_pred, y_true)) # must be (0,1) rangee
    
    return (msssim_w*msssim_value) + (l1_w*l1_value)