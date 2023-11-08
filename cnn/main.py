import cv2
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from ultralytics import YOLO

from dataset import V2Dataset
from train_val import train, val 
from hands_cnn import Hands_CNN
from face_cnn import Face_CNN

# Place the trainable model classes here so we can initialize them from this script
available_models = {
                    'hands' : Hands_CNN, 
                    'face' : Face_CNN
                    }

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

    if args.model in available_models:
        model = available_models[args.model](args, out_features=out_features)
        model_name = args.model
    else:
        raise ValueError(f'Model Not Supported: {args.model}')

    model.to(args.device)

    detector = YOLO(args.detector_path)
    detector = detector.to(args.device)

    optimizer = optimizer_type(args.optimizer, model, args.lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        loss, _ = train(model, detector, args.device, train_dataloader, optimizer, criterion, epoch, model_name=model_name)
        if epoch % 5 == 0: val(model, detector, args.device, test_dataloader, criterion, epoch, model_name=model_name)

        if loss < best_loss and epoch % args.save_period == 0:
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
    args.add_argument('--epochs', type=int, default=30)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--num_frozen_params', type=int, default=30)
    args.add_argument('--save_period', type=int, default=5)
    args.add_argument('--transform', type=bool, default=True) 
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--data_dir', type=str, default='data/v2_cam1_cam2_split_by_driver')
    args.add_argument('--model_dir', type=str, default='cnn/hands_models')
    args.add_argument('--detector_path', type=str, default='path/to/yolo/weights')
    args.add_argument('--optimizer', type=str, default='Adam')
    args.add_argument('--scheduler', action='store_true')
    args.add_argument('--model', type=str, default='hands')

    args = args.parse_args()

    run_main(args)


        
