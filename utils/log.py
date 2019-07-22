import torch
from torchvision import utils
import time
import datetime

def log_time(msg):
	print(msg)
	print('\t', end='')
	print('Datetime: {}'.format(datetime.datetime.now()), end='\n')

def log_images(x, y, out, path):
	frames = torch.split(x, 1, dim=2)
	frames = [frame.squeeze(dim=2) for frame in frames]
	#print(frames[0].shape)
	#print(frames[1].shape)

	frames.append(out)
	frames.append(y)

	grid = utils.make_grid(frames, padding=100)
	utils.save_image(grid, path)

def log_model_eval(model):
	print('Model evaluation: ', model.eval())

def log_model_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Model trainable parameters: ', params)
