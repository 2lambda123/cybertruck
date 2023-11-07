import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
import argparse
from dataset import V1Dataset, V2Dataset
from ultralytics import YOLO
from cnn.train_val import *

class Hands_CNN(nn.Module):
    def __init__(self, args, out_features=10):
        super(Hands_CNN, self).__init__()

        feature_extractor = vgg16(weights=VGG16_Weights.DEFAULT).features
        if args.freeze: feature_extractor = self.freeze(feature_extractor)  

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
    
    def freeze(self, feature_extractor):
        for param in feature_extractor.parameters():
            param.requires_grad = False
        return feature_extractor
    

    def forward(self, x):
        return self.model(x)
    

def run_main(args):

    if args.transform:
        transform = transforms.Compose([
            transforms.Resize((640,640)),
            transforms.ToTensor(),
        ])

    train_dataset = V2Dataset(cam1_path='data/v2_cam1_cam2_ split_by_driver/Camera 1/train', cam2_path='data/v2_cam1_cam2_ split_by_driver/Camera 2/train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = V2Dataset(cam1_path='data/v2_cam1_cam2_ split_by_driver/Camera 1/test', cam2_path='data/v2_cam1_cam2_ split_by_driver/Camera 2/test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    out_features = len(train_dataset.classes)
    model = Hands_CNN(args, out_features=out_features)
    model.to(args.device)

    detector = YOLO(args.detector_path)
    detector = detector.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        loss, _ = train(model, detector, args.device, train_dataloader, optimizer, criterion, epoch, args.batch_size)
        if epoch % 5: val(model, detector, args.device, test_dataloader, epoch)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f'{args.model_dir}/best_hands_cnn.pt')
            print(f'Saved model at epoch {epoch}')



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--hidden_units', type=int, default=128)
    args.add_argument('--freeze', type=bool, default=True)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--epochs', type=int, default=4)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--transform', type=bool, default=True) 
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--data_dir', type=str, default='data')
    args.add_argument('--model_dir', type=str, default='cnn/models')
    args.add_argument('--detector_path', type=str, default='/home/ron/Classes/CV-Systems/cybertruck/hands_detection/runs/detect/mds_model_w_aug(best)/weights/best.pt')
    args.add_argument('--optimizer', type=str, default='Adam')
    args.add_argument('--loss', type=str, default='CrossEntropyLoss')
    args.add_argument('--scheduler', action='store_true')

    args = args.parse_args()

    run_main(args)


        
