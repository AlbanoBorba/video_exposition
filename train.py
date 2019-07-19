import time
import datetime
import torch

def log_time(name, video_step, sample_step):
	print('Video Step: {} | Sample Step: {}'.format(video_step, sample_step))
	print('\t', end='')
	print(datetime.datetime.now())

def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model.train()  # Set model to training mode

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		print('Datetime: {}'.format(datetime.datetime.now()), end='\n')
				
		running_loss = 0.0
		n_samples = 0

		# Iterate over videos.
		for video_step, video_loader in enumerate(dataloader.iterate()):
			# Iterate over frames.
			for sample_step, sample in  enumerate(video_loader):
				n_samples = sample_step * video_step                                                
				
				# send data to device
				y, x = sample['y'].to(device), sample['x'].to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				outputs = model(x)
				
				# loss
				loss = criterion(outputs, y)

				# backward + optimize
				loss.backward()
				optimizer.step()
					
				# statistic
				running_loss += loss.data

			if n_samples % 20000 == 0:
				print('Running Loss: {:.4f}, Sample Loss: {:.4f}'.format(running_loss, running_loss / n_samples), end='\n')
				# save image

		#save model    
		torch.save(model.state_dict(), './3dcnn_weigths_{}.pth'.format(epoch))
		epoch_loss = running_loss / n_samples

		print('Running Loss: {:.4f}, Epoch Loss: {:.4f}'.format(running_loss, epoch_loss), end='\n\n')

	return model
