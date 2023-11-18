from detection.cnn.dataset import V2Dataset

import torch

from torchvision.models.resnet import ResNet, Bottleneck, resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch import nn

def train_one_epoch(model: ResNet, device: torch.device, input: torch.Tensor, target: int, criterion, epoch: int):
  if not model.training:
    raise Exception("Model must be in training mode in order to train.")

  
  # Forward pass of input
  output = model(input)

  # Compute loss
  loss = criterion(output, target)

  # Backward pass
  loss.backward()

  return loss


if __name__ == '__main__':
  model = resnet50(num_classes=10)
  model.train()
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model.to(device)

  transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
  ])

  criterion = nn.CrossEntropyLoss()

  train_dataset = V2Dataset(cam1_path="./data/v2_cam1_cam2_split_by_driver/Camera 1", cam2_path="./data/v2_cam1_cam2_split_by_driver/Camera 2", transform=transform)
  train_dataloader = DataLoader(train_dataset, batch_size=64)

  epochs = 100

  for epoch in tqdm(range(epochs)):
    for batch in tqdm(train_dataloader):
      input, target = batch

      input, target = input.to(device), target.to(device)

      loss = train_one_epoch(model=model, device=device, input=input, target=target, criterion=criterion, epoch=epoch)

    # Log statistics
    print(f"Epoch {epoch} loss = {loss.item}")



  




  
  