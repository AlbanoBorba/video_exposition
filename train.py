import time
import datetime
import torch

def train_model(model, dataloader, criterion, optimizer, num_epochs=25, phases=['train', 'val']):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ('train' not in phases) and ('val' not in phases):
        print("O parâmetro <phases> deve ser 'train' ou 'val'.")

    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #print(datetime.datetime.now(), end='\n')

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            n_samples = 0

            # Iterate over data.
            for video_step, video_loader in enumerate(dataloader.iterate()):
                for sample_step, sample in  enumerate(video_loader):
                    print(datetime.datetime.now(), end='\n')
                    print('Video Step: {} | Sample Step: {}'.format(video_step, sample_step))
                    n_samples = sample_step * video_step                                
                    #print(step)
                    #print(sample)

                    y, x = sample['y'].to(device), sample['x'].to(device)
                    #print(x)                                   
                    #print('Shape X')
                    #print(x.shape)

                    #print('Shape Y')
                    #print(y.shape)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # Get model outputs and calculate loss
                        outputs = model(x)
                        loss = criterion(outputs, y)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statisti
                    running_loss += loss.item()
                    torch.cuda.empty_cache()

            #save model    
            torch.save(model.state_dict(), '/3dcnn_weiths_{}.pth'.format(epoch))


            epoch_loss = running_loss / len(n_samples)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Running Loss: {:.4f}, Epoch Loss: {:.4f}'.format(phase, running_loss, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_histor