import os
import cv2
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import v2
from ultralytics import YOLO
from torch.nn import DataParallel as DP
from tqdm import tqdm

from cnn.train_val import train, val
from cnn.dataset import V2Dataset
from cnn.hands_cnn import Hands_VGG16, Hands_Efficient, Hands_Squeeze, Hands_InceptionV3
from cnn.face_cnn import Face_CNN
from wrappers.hands_wrapper import Hands_Inference_Wrapper


class Ensemble(nn.Module):
    def __init__(self, models, num_classes):
        (Ensemble, self).__init__()
        num_models = len(models)
        self.models = nn.ModuleList([self.freeze(model) for model in models])
        # self.weights = nn.Parameter(torch.ones(num_models))
        self.classifier = nn.Linear(num_classes * num_models, num_classes)

    def freeze(self, model):
        for param in list(model.parameters()):
            param.requires_grad = False
        return model

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        
        # # Currently not Using this.
        # # TODO: use a genetic algorithm to determine the weights
        # weighted_outputs = [output * weight for output, weight in zip(outputs, self.weights)]
        # genetic_alg = 

        stacked_outputs = torch.cat(outputs, dim=-1)
        output = self.classifier(stacked_outputs)
        return output


def optimizer_type(args, model):
    if args.optimizer == 'Adam' or args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW' or args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD' or args.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer not supported')

def select_model_and_start(args, out_features):

    hands_cnn = Hands_VGG16(args, out_features=out_features)
    face_cnn = Face_CNN(args, out_features=out_features)

    cnns = [Hands_Inference_Wrapper(hands_cnn), face_cnn]

    model = Ensemble(args, cnns)

    if args.distributed and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = DP(model)

    if args.resume_path is not None:
        model.load_state_dict(torch.load(args.resume_path))
        print(f'Resuming from {args.resume_path}')

        if args.resume_last_epoch:
            epoch_start = int(args.resume_path.split('/')[-1].split('_')[0].split('epoch')[-1])
        else: epoch_start = 1
    
    return model, epoch_start


def run_main(args):

    # model_names = ['hands_vgg', 'face', 'raw']
    # model_input_size = {
    #                     'hands_vgg' : 640,
    #                     'face' : 640,
    #                     'raw' : 224
    #                     }

    # needed for dynamic resizing of images
    # def custom_collate(batch, model_name):
    #     resize = transform((model_input_size[model_name], model_input_size[model_name]))
    #     return [(resize(sample[0]), sample[1]) for sample in batch]

    train_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/train', cam2_path=f'{args.data_dir}/Camera 2/train')
    val_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/test', cam2_path=f'{args.data_dir}/Camera 2/test')

    #ensures that the shuffling is the same for all models
    # train_shuffled_idxs = torch.randperm(len(train_dataset))
    # val_shuffled_idxs = torch.randperm(len(train_dataset))


    # train_loader = {
    #     model_name: DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_shuffled_idxs),
    #                            collate_fn=lambda batch: custom_collate(batch, model_name))
    #     for model_name in model_names 
    # }

    # val_loader = { model_name: DataLoader(val_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_shuffled_idxs),
    #                            collate_fn=lambda batch: custom_collate(batch, model_name)) 
    #                for model_name in model_names 
    # }


    # Can't pass transform to dataloader because some CNNs need different initial resizing. For now, doing it in wrapper class.
    train_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/train', cam2_path=f'{args.data_dir}/Camera 2/train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/test', cam2_path=f'{args.data_dir}/Camera 2/test')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    out_features = len(train_dataset.classes)

    model, model_name, epoch_start = select_model_and_start(args, out_features)

    model.to(args.device)
    print(model)

    optimizer = optimizer_type(args, model)  
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_start, args.epochs + 1):
        loss, _ = train(model, None, args.device, train_loader, optimizer, criterion, epoch, model_name=model_name)
        if epoch % 5 == 0: val(model, None, args.device, val_loader, criterion, epoch, model_name=model_name)
    pass

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--distributed', type=bool, default=False)

    args.add_argument('--resume_path', type=str, default=None)
    args.add_argument('--resume_last_epoch', type=bool, default=False)

    args.add_argument('--hidden_units', type=int, default=128)
    args.add_argument('--freeze', type=bool, default=True)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--epochs', type=int, default=30)

    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--dropout', type=float, default=0.5)
    args.add_argument('--optimizer', type=str, default='Adam')
    args.add_argument('--weight_decay', type=float, default=0.0)
    args.add_argument('--scheduler', action='store_true')

    args.add_argument('--save_period', type=int, default=5)
    args.add_argument('--transform', type=bool, default=True) 
    args.add_argument('--device', type=str, default='cuda')

    args.add_argument('--data_dir', type=str, default='data/v2_cam1_cam2_split_by_driver')
    args.add_argument('--model_dir', type=str, default='cnn/hands_models')
    args.add_argument('--detector_path', type=str, default='path/to/yolo/weights')

    args = args.parse_args()

    run_main(args)