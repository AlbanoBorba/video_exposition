import time
import datetime
import torch

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print(datetime.datetime.now(), end='\n')
        
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        n_samples = 0

        # Iterate over data.
        for video_step, video_loader in enumerate(dataloader.iterate()):
            for sample_step, sample in  enumerate(video_loader):
                n_samples = sample_step * video_step                                                
                
                print('Video Step: {} | Sample Step: {}'.format(video_step, sample_step))
                print('\t', end='')
                print(datetime.datetime.now())

                y, x = sample['y'].to(device), sample['x'].to(device)
              
                print('To device')
                print('\t', end='')
                print(datetime.datetime.now())

                # zero the parameter gradients
                optimizer.zero_grad()
                print('Zero grad')
                print('\t', end='')
                print(datetime.datetime.now())

                # forward
                outputs = model(x)
                print('Forward')
                print('\t', end='')
                print(datetime.datetime.now())
                
                # loss
                loss = criterion(outputs, y)
                print('Loss')
                print('\t', end='')
                print(datetime.datetime.now())

                # backward + optimize
                print('Backward + optimize')
                print('\t', end='')
                print(datetime.datetime.now()) 
                loss.backward()
                optimizer.step()
                    
                # statistic
                running_loss += loss.data
                print('Statistic')
                print('\t', end='')
                print(datetime.datetime.now())
                #torch.cuda.empty_cache()

        #save model    
        torch.save(model.state_dict(), '/3dcnn_weiths_{}.pth'.format(epoch))


        epoch_loss = running_loss / n_samples
        #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('{} Running Loss: {:.4f}, Epoch Loss: {:.4f}'.format(running_loss, epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model