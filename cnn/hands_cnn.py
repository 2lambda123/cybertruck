import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import argparse
from dataset import V1Dataset, V2Dataset
from ultralytics import YOLO
from train_test import *

class Hands_CNN(nn.Module):
    def __init__(self, args, out_features=10):
        super(Hands_CNN, self).__init__()

        feature_extractor = vgg16(pretrained=True).features
        if args.freeze: feature_extractor = self.freeze(vgg16(pretrained=True).features)  

        in_features = feature_extractor[-3].out_channels 
        classifier = nn.Sequential(
            nn.Linear(in_features, args.hidden_units),
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

        self.detector = YOLO(args.detector_path)
    
    def freeze(self, feature_extractor):
        for param in feature_extractor.parameters():
            param.requires_grad = False
        return feature_extractor
    
    def extract_detection(self, results):
        for result in results:
            boxes = results.boxes
        # TODO extact boxes from results create images from boxes
        # use transform to convert back images to tensor and resize((224,224)))
        # use torch.stack to stack all images back onto batch
        pass

    def forward(self, x):
        results = self.detector(x)
        x = self.extract_detection(results)
        return self.model(x)
    

def run_main(args):

    if args.transform:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    train_dataset = V2Dataset(cam1_path='data/v2_cam1_cam2_ split_by_driver/Camera 1/train', cam2_path='data/v2_cam1_cam2_ split_by_driver/Camera 2/train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = V2Dataset(cam1_path='data/v2_cam1_cam2_ split_by_driver/Camera 1/test', cam2_path='data/v2_cam1_cam2_ split_by_driver/Camera 2/test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    out_features = len(train_dataset.classes)
    model = Hands_CNN(args, out_features=out_features)
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, args.epochs):
        train(model, args.device, train_dataloader, optimizer, criterion, epoch, args.batch_size)

    test(model, args.device, test_dataloader)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--hidden_units', type=int, default=128)
    args.add_argument('--freeze', type=bool, default=True)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--epochs', type=int, default=30)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--transform', type=bool, default=True) 
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--data_dir', type=str, default='data')
    args.add_argument('--model_dir', type=str, default='model')
    args.add_argument('--detector_path', type=str, default='/home/ron/Classes/CV-Systems/cybertruck/hands_detection/runs/detect/mds_model_w_aug(best)/weights/best.pt')
    args.add_argument('--optimizer', type=str, default='Adam')
    args.add_argument('--loss', type=str, default='CrossEntropyLoss')
    args.add_argument('--scheduler', action='store_true')

    args = args.parse_args()

    run_main(args)


        
