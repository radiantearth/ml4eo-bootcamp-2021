import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from PIL import Image

def train_model_snapshot(model, criterion, lr, dataloaders, dataset_sizes, device, num_cycles, num_epochs_per_cycle):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000000.0
    model_w_arr = []
    for cycle in range(num_cycles):
        #initialize optimizer and scheduler each cycle
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 10*len(dataloaders['train']))
        for epoch in range(num_epochs_per_cycle):
            print('Cycle {}: Epoch {}/{}'.format(cycle, epoch, num_epochs_per_cycle - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, inputs_area, inputs_mask, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    inputs_mask = inputs_mask.to(device)
                    inputs_area = inputs_area.to(device)
                    labels = labels.to(device)
                    

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, inputs_mask)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()
        # deep copy snapshot
        model_w_arr.append(copy.deepcopy(model.state_dict()))

    ensemble_loss = 0.0

    #predict on validation using snapshots
    for inputs, inputs_area, inputs_mask, labels in dataloaders['val']:
        inputs = inputs.to(device)
        inputs_mask = inputs_mask.to(device)
        inputs_area = inputs_area.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        prob = torch.zeros((inputs.shape[0], 7), dtype = torch.float32).to(device)
        for weights in model_w_arr:
            model.load_state_dict(weights)
            model.eval()
            outputs = model(inputs, inputs_mask)
            prob += F.softmax(outputs, dim = 1)
        
        prob /= num_cycles
        loss = F.nll_loss(torch.log(prob), labels)    
        ensemble_loss += loss.item() * inputs.size(0)
    
    ensemble_loss /= dataset_sizes['val']

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Ensemble Loss : {:4f}, Best val Loss: {:4f}'.format(ensemble_loss, best_loss))

    # load snapshot model weights and combine them in array
    model_arr =[]
    for weights in model_w_arr:
        model.load_state_dict(weights)   
        model_arr.append(model) 
    
    return model_arr, ensemble_loss, best_loss

def test(models_arr, loader, device):
    res = np.zeros((1402, 7), dtype = np.float32)
    for model in models_arr:
        model.eval()
        res_arr = []
        for inputs, inputs_area, inputs_mask, _ in loader:
            inputs = inputs.to(device)
            inputs_mask = inputs_mask.to(device)
            inputs_area = inputs_area.to(device)
            # forward
            with torch.set_grad_enabled(False):
                outputs = F.softmax(model(inputs, inputs_mask), dim = 1)    
                res_arr.append(outputs.detach().cpu().numpy())
        res_arr = np.concatenate(res_arr, axis = 0)
        
        res += res_arr
    return res / len(models_arr)

def load_file(fp):
    """Takes a PosixPath object or string filepath
    and returns np array"""
    
    return np.array(Image.open(fp.__str__()))
