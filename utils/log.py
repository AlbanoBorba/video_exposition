import torch
from torchvision import utils
import time
import datetime
import numpy as np

def log_time(msg):
	print(msg)
	print('\t', end='')
	print('Datetime: {}'.format(datetime.datetime.now()), end='\n')

def log_images(x, y, out, path):
	print('Log:')
	print(x.shape)
	frames = torch.split(x, 1, dim=1)
	frames = [frame.squeeze(dim=1) for frame in frames]
	print(frames[0].shape)
	frames.append(out)
	frames.append(y)
	
	#frames = torch.stack([out, y], dim=1)
	# frames = torch.cat([x, s], dim=1)

	#grid = utils.make_grid(frames)
	#print(grid.shape)
	
	count = 0
	for frame in frames:
		grid = utils.make_grid(frame)
		utils.save_image(grid, path + 'frame_{}.png'.format(count))
		count += 1

def log_model_eval(model):
	print('Model evaluation: ', model.eval())

def log_model_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Model trainable parameters: ', params)
