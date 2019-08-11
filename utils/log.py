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
	
	with torch.no_grad():
		frames = torch.split(x.cpu(), 1, dim=2)
		frames = [frame.squeeze(dim=2) for frame in frames]
		frames.append(out.cpu())
		frames.append(y.cpu())

		#frames = torch.cat(frames, dim=3)
		frames = [f.squeeze() for f in frames]

		grid = utils.make_grid(frames)
		utils.save_image(grid, path + 'sample.png')

def log_model_eval(model):
	print('Model evaluation: ', model.eval())

def log_model_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Model trainable parameters: ', params)

def log_config(save_path, **kwargs):
	
	with open('{}config.txt', "w") as file:
		file.write('Datetime: {}'.format(datetime.datetime.now()), end='\n\n')
		for key, value kwargs.items():
			file.write('{}: {}'.format(key, value))
