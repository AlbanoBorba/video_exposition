import time
import datetime
import torch

def test_model(model, data, criterion, optimizer)
	pass

def train_model(model, data, criterion, optimizer):
				
	model.train()  # Set model to training mode

	# zero the parameter gradients
	optimizer.zero_grad()

	# forward
	outputs = model(x)
	
	# loss
	loss = criterion(outputs, y)

	# backward + optimize
	loss.backward()
	optimizer.step()
		
	return loss.data