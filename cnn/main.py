import os
import cv2
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from ultralytics import YOLO
from torch.nn import DataParallel as DP

from dataset import V2Dataset
from train_val import train, val 
from hands_cnn import Hands_VGG16
from face_cnn import Face_CNN
from raw_cnn import Raw_CNN

# Place the trainable model classes here so we can initialize them from this script
available_models = {
                    'face' : Face_CNN,
                    'hands_vgg' : Hands_VGG16,
                    'raw': Raw_CNN
                   }

def optimizer_type(args, model):
    if args.optimizer == 'Adam' or args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW' or args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD' or args.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer not supported')

def select_model_and_start(args, num_classes):
    ''' 
    Selects the model, initializes it with the correct number of classes,
    and returns the model and model name.
    '''
    epoch_start = 1

    if args.model in available_models:
        model = available_models[args.model](args, num_classes=num_classes)
        model_name = args.model
    else:
        raise ValueError(f'Model Not Supported: {args.model}')

    # if we want to use multiple GPUs, we need to wrap the model in a DataParallel object.
    # not working as expected on newton
    if args.distributed and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = DP(model)

    # if we want to resume training, load the model from the checkpoint
    if args.resume_path is not None:
        model.load_state_dict(torch.load(args.resume_path))
        print(f'Resuming from {args.resume_path}')
        # if we want to resume from the last epoch, we need to parse the epoch number from the checkpoint name
        if args.resume_last_epoch:
            epoch_start = int(args.resume_path.split('/')[-1].split('_')[0].split('epoch')[-1])
    
    return model, model_name, epoch_start

def transform_type(model_name):

    if model_name != 'raw':
        train_transform = v2.Compose([
            v2.Resize((640, 640)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        val_transform = v2.Compose([
            v2.Resize((640, 640)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
    else:
        train_transform = v2.Compose([
            v2.ToPILImage(),
            v2.Resize((299, 299)),
            v2.RandomHorizontalFlip(p=0.4),
            v2.RandomPerspective(distortion_scale=0.1, p=0.25),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.3102, 0.3102, 0.3102], std=[0.3151, 0.3151, 0.3151])
        ])

        val_transform = v2.Compose([
            v2.ToPILImage(),
            v2.Resize((299, 299)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.3879, 0.3879, 0.3879], std=[0.3001, 0.3001, 0.3001])
        ])

    return train_transform, val_transform

def run_main(args):
    # select the correct transform based on the model. Selected 640x640 because that is the input size of the detector. Can take any input though. 
    train_transform, val_transform = transform_type(args.model)

    train_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/train', cam2_path=f'{args.data_dir}/Camera 2/train', transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/test', cam2_path=f'{args.data_dir}/Camera 2/test', transform=val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    num_classes = len(train_dataset.classes)

    model, model_name, epoch_start = select_model_and_start(args, num_classes)

    model.to(args.device)
    print(model)

    # initialize the detector
    if model_name == "raw":
        detector = None
    else: 
        detector = YOLO(args.detector_path)
        detector.to(args.device)

        
    optimizer = optimizer_type(args, model)  
    criterion = nn.CrossEntropyLoss()

    # if we want to use a learning rate scheduler, initialize it here
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.5) if args.scheduler else None

    # initialize the best loss to infinity so that the first loss is always better
    best_loss = np.inf
    # train for a given set of epochs, run validation every five, and save the model if the loss is the best so far
    for epoch in range(epoch_start, args.epochs + 1):
        loss, _ = train(model, detector, args.device, train_dataloader, optimizer, criterion, epoch, model_name=model_name, scheduler=scheduler)
        if epoch % 5 == 0: val(model, detector, args.device, test_dataloader, criterion, epoch, model_name=model_name)

        if loss < best_loss and epoch % args.save_period == 0:
            best_loss = loss

            now = datetime.now()
            time_now = now.strftime('%m-%d_%H:%M:%S')

            model_type = args.model.split('_')[-1]
            save_dir =  os.path.join(args.model_dir, model_type, args.optimizer)

            os.makedirs(save_dir, exist_ok=True)

            torch.save(model.state_dict(), f'{save_dir}/epoch{epoch}_{time_now}.pt')
            print(f'Saved model at epoch {epoch}')



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='hands_inception')
    args.add_argument('--distributed', type=bool, default=False)

    args.add_argument('--resume_path', type=str, default=None)
    args.add_argument('--resume_last_epoch', type=bool, default=False)

    args.add_argument('--hidden_units', type=int, default=128)
    args.add_argument('--freeze', type=bool, default=True)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--epochs', type=int, default=30)

    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--num_frozen_params', type=int, default=30)
    args.add_argument('--dropout', type=float, default=0.5)
    args.add_argument('--optimizer', type=str, default='Adam')
    args.add_argument('--weight_decay', type=float, default=0.0)

    args.add_argument('--scheduler', action='store_true')
    args.add_argument('--scheduler_step_size', type=int, default=10)

    args.add_argument('--save_period', type=int, default=5)
    args.add_argument('--transform', type=bool, default=True) 
    args.add_argument('--device', type=str, default='cuda')

    args.add_argument('--data_dir', type=str, default='data/v2_cam1_cam2_split_by_driver')
    args.add_argument('--model_dir', type=str, default='cnn/hands_models')
    args.add_argument('--detector_path', type=str, default='path/to/yolo/weights')

    args = args.parse_args()

    run_main(args)


        

