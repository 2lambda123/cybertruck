from detection.cnn.dataset import V2Dataset

import torch

from torchvision.models.resnet import ResNet, Bottleneck, resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch import nn, optim
import numpy as np

def train_one_epoch(model: ResNet, device: torch.device, criterion, data_loader: DataLoader, optimizer):
  if not model.training:
    raise Exception("Model must be in training mode in order to train.")
  
  losses = []
  correct, total = 0, 0

  for batch in tqdm(data_loader):
    input, target = batch

    input, target = input.to(device), target.to(device)

    optimizer.zero_grad()

    # Forward pass of input
    output = model(input)

    # Compute loss
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    losses.append(loss.item())

    optimizer.step()

    pred = output.argmax(dim=1, keepdim=True)
    total += len(target)

    correct += pred.eq(target.view_as(pred)).sum().item()

  train_loss = float(np.mean(losses))
  train_acc = (correct / total) * 100

  return train_loss, train_acc

def val(model: ResNet, device: torch.device, test_loader: DataLoader, criterion, epoch: int):
  if model.training:
    raise Exception("Model must be in eval mode in order to validate.")
  
  losses = []
  correct, total = 0, 0

  with torch.no_grad():
    for sample in tqdm(test_loader):
      input, target = sample
      input, target = input.to(device), target.to(device)

      output = model(input)
      
      loss = criterion(output, target)
      losses.append(loss.item())

      pred = output.argmax(dim=1, keepdim=True)
      total += len(target)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss = float(np.mean(losses))
  accuracy = 100. * (correct / total)

  print(f'==========================Validation at epoch {epoch}==========================')
  print(f'\nAverage loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')
  print(f'===============================================================================')
  return test_loss, accuracy

if __name__ == '__main__':
  model = resnet50(num_classes=10)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.train()

  transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
  ])

  criterion = nn.CrossEntropyLoss()

  train_dataset = V2Dataset(cam1_path="./data/v2_cam1_cam2_split_by_driver/Camera 1/train", cam2_path="./data/v2_cam1_cam2_split_by_driver/Camera 2/train", transform=transform)
  train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

  test_dataset = V2Dataset(cam1_path="./data/v2_cam1_cam2_split_by_driver/Camera 1/test", cam2_path="./data/v2_cam1_cam2_split_by_driver/Camera 2/test", transform=transform)
  test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  epochs = 100

  for epoch in tqdm(range(epochs)):
    loss, train_acc = train_one_epoch(model=model, device=device, data_loader=train_dataloader, criterion=criterion, optimizer=optimizer)  

    if (epoch + 1) % 5 == 0:
      model.eval()
      val(model=model, device=device, test_loader=test_dataloader, criterion=criterion, epoch=epoch)      
      model.train()

    if (epoch + 1) % 10 == 0:
      torch.save(model.state_dict(), f"general-detect-{epoch}.pt")

    # Log statistics
    print(f"Epoch {epoch} loss = {float(np.mean(loss))}, train accuracy = {train_acc}")



  




  
  