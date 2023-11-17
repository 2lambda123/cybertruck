import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

from face_cnn import extract_face_detections
from hands_cnn import extract_hands_detection         
        


def train(model, detector, device, train_loader, optimizer, criterion, epoch, model_name="hands_vgg", scheduler=None):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct, total = 0, 0
    
    
    # Iterate over entire training samples in batches
    for batch_sample in tqdm(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)

        # detector would be None during ensemble training since it would be encapsulated by model wrapper classes
        if detector is not None:
            detections = detector(data, verbose=False)

            if model_name == 'face':
                data = extract_face_detections(detections)
                data = model.preprocessor(data)
                data.to(device)
            else: 
                # The default hands cnn extraction
                data, target = extract_hands_detection(data, detections, target, model_name, train_mode=True)


        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        if model_name == 'hands_inception':
            output = model(data)[0]
        else:
            output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(output,target)

        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()

        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        total += len(target)
        # ======================================================================
        # Count correct predictions overall 
        correct += pred.eq(target.view_as(pred)).sum().item()

    if scheduler is not None:
            scheduler.step()  

    train_loss = float(np.mean(losses))
    train_acc = (correct / total) * 100.
    print(f'Epoch {epoch:03} - Average loss: {float(np.mean(losses)):.4f}, Accuracy: {correct}/{total} ({train_acc:.2f}%)\n')
    return train_loss, train_acc
    


def val(model, detector, device, test_loader, criterion, epoch, model_name="hands_vgg"):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct, total = 0, 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for  sample in tqdm(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            
            
            # detector would be None during ensemble training since it would be encapsulated by model wrapper classes   
            if detector is not None:
                detections = detector(data, verbose=False)

                if model_name == 'face':
                    data = extract_face_detections(detections)
                    data = model.preprocessor(data)
                    data.to(device)
                else: 
                    # The default hands cnn extraction
                    data, target = extract_hands_detection(data, detections, target, model_name, train_mode=False)

            
            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute loss based on same criterion as training 
            loss = criterion(output,target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            total += len(target)
            # ======================================================================
            # Count correct predictions overall 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = (correct / total) * 100.

    print(f'==========================Validation at epoch {epoch}==========================')
    print(f'\nAverage loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    print(f'===============================================================================')
    return test_loss, accuracy


# TODO, replace with tensorboard.
def plot_accuracies(train_acc, test_acc, mode, x_len):
    '''
    Plots the accuracies
    train_acc: Array-like. Contains all of training accuracies
    test_acc: Array-like. Contains all of the test accuracies
    mode: str. Needed to add to title.
    x_len: int. Number of epochs.  Used to define length of x-axis.
    '''
    fig = plt.figure(figsize=(15, 5))
    x_axis = np.linspace(1,x_len,x_len)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_axis, train_acc,'b-',zorder=1,label='Training')
    ax.scatter(x_axis, train_acc,c='white',edgecolors='blue',zorder=2)

    ax.plot(x_axis, test_acc,'r-',zorder=1,label='Test')
    ax.scatter(x_axis, test_acc,c='white',edgecolors='red',zorder=2)
    ax.set_title("Model " + mode)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc=4)
    ax.grid(alpha=0.05)