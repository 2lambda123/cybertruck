import numpy as np
import tqdm as tqdm
import torch
import torch.nn as nn


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0

    # output_list = []


    for batch_idx, batch in enumerate(tqdm(train_loader)):

        data, target = batch


        # print('data', data)
        # print('target', target)

        # Push data/label to correct device
        data, text = data.to(device=device,non_blocking=True), target.to(device=device,non_blocking=True)
        
        # print(f'data: {data.shape}')
        # print(f'text: {text.shape}')

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        loss = criterion(data, text)
        # loss.mean().requires_grad = True
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())   
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # ======================================================================
        # Count correct predictions overall 
        correct += pred.eq(target.view_as(pred)).sum().item()


        if batch_idx % 100 == 0:
            print(f'batch_idx: {batch_idx}, current loss: {float(np.mean(losses))}')
        
    train_loss = float(np.mean(losses))
    # train_acc = correct / ((batch_idx+1) * batch_size)
    # print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
    #     100. * correct / ((batch_idx+1) * batch_size)))
    print(f'Train set: Average loss: {train_loss}')
    
    torch.save(model.state_dict(), f'MODEL/model_{epoch}.pth')

    return train_loss