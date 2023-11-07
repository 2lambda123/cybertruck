import cv2
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from ultralytics import YOLO

from dataset import V1Dataset, V2Dataset
from train_val import *

class Hands_CNN(nn.Module):
    def __init__(self, args, out_features=10):
        super(Hands_CNN, self).__init__()

        feature_extractor = vgg16(weights=VGG16_Weights.DEFAULT).features
        if args.freeze: feature_extractor = self.freeze(feature_extractor, args.num_frozen_params)  

        in_features = feature_extractor[-3].out_channels 
        classifier = nn.Sequential(
            nn.Linear(in_features * 7 * 7, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, out_features),
        )

        self.model = nn.Sequential(
            feature_extractor,
            nn.Flatten(),
            classifier,
        )
    
    def freeze(self, feature_extractor, num_frozen_params):
        for param in list(feature_extractor.parameters())[: num_frozen_params]:
            param.requires_grad = False
        return feature_extractor
    

    def forward(self, x):
        return self.model(x)
    

def optimizer_type(optimizer, model, lr):
    if optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError('Optimizer not supported')

def run_main(args):

    if args.transform:
        transform = transforms.Compose([
            transforms.Resize((640,640)),
            transforms.ToTensor(),
        ])

    train_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/train', cam2_path=f'{args.data_dir}/Camera 2/train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/test', cam2_path=f'{args.data_dir}/Camera 2/test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    out_features = len(train_dataset.classes)
    model = Hands_CNN(args, out_features=out_features)
    model.to(args.device)

    detector = YOLO(args.detector_path)
    detector = detector.to(args.device)

    optimizer = optimizer_type(args.optimizer, model, args.lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        loss, _ = train(model, detector, args.device, train_dataloader, optimizer, criterion, epoch)
        if epoch % 5 == 0: val(model, detector, args.device, test_dataloader, criterion, epoch)

        if loss < best_loss:
            best_loss = loss

            now = datetime.now()
            time_now = now.strftime('%m-%d(%H:%M:%S)')

            torch.save(model.state_dict(), f'{args.model_dir}/{args.optimizer}/epoch{epoch}_{time_now}.pt')
            print(f'Saved model at epoch {epoch}')



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--hidden_units', type=int, default=128)
    args.add_argument('--freeze', type=bool, default=True)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--epochs', type=int, default=4)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--num_frozen_params', type=int, default=30)
    args.add_argument('--transform', type=bool, default=True) 
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--data_dir', type=str, default='data/v2_cam1_cam2_split_by_driver')
    args.add_argument('--model_dir', type=str, default='cnn/hands_models')
    args.add_argument('--detector_path', type=str, default='path/to/yolo/weights')
    args.add_argument('--optimizer', type=str, default='Adam')
    args.add_argument('--scheduler', action='store_true')

    args = args.parse_args()

    run_main(args)


        
