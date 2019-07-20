from torchvision import utils

def log_time(msg):
	print(msg)
	print('\t', end='')
    print('Datetime: {}'.format(datetime.datetime.now()), end='\n')

def log_images(x, y, path):
	utils.save_image(x, path+'x.png')
	utils.save_image(y, path+'y.png')

def log_model_eval(model):
	print('Model evaluation: ', model.eval())

def log_model_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Model trainable parameters: ', params)
