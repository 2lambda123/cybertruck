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
import torch.nn.functional as F
from tqdm import tqdm

from cnn.dataset import V2Dataset
from cnn.hands_cnn import Hands_VGG16
from cnn.face_cnn import Face_CNN
from cnn.raw_cnn import Raw_CNN
from wrappers.hands_wrapper import Hands_Inference_Wrapper
from wrappers.face_wrapper import Face_Inference_Wrapper



class Ensemble(nn.Module):
    def __init__(self, args, models, train_loader, val_loader, num_classes=10):
        super(Ensemble, self).__init__()
        self.args = args
        self.device = args.device

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        num_models = len(models)

        self.ensemble = nn.ModuleList([model for model in models])
        self.set_weights = torch.tensor([0.45, 0.15, 0.45])
        self.weights = nn.Parameter(torch.ones(num_models) / num_models, requires_grad=True)


    def save_last_learnable_param(self, file_path):
        # Create a dictionary to store the state of only the last learnable parameter
        save_dict = {'weight': self.weights.state_dict()}

        # Save the dictionary to a file
        torch.save(save_dict, file_path)

    def load_last_learnable_param(self, file_path):
        # Load the dictionary from the file
        save_dict = torch.load(file_path)

        # Set requires_grad to True for the last learnable parameter ('weight')
        last_learnable_param_name = 'weight'
        if last_learnable_param_name in save_dict:
            self.weights.load_state_dict(save_dict[last_learnable_param_name])
            self.weights.requires_grad = True

    def set_to_train(self, train=True):
        for model in self.ensemble:
            for name, param in model.named_parameters():
                if train:
                    if param.requires_grad == True and name != 'weight':
                        param.requires_grad = False
                        self.weights.requires_grad = True
                else:
                    self.weights.requires_grad = True



    def _custom_forward(self, x, training=True):   
        self.set_to_train(training)

        outputs = [model(x) for model in self.ensemble]
        weighted_preds = [output * pred * set_weight for output, pred, set_weight in zip(outputs, self.weights, self.set_weights)]

        stacked_outputs = torch.stack(weighted_preds, dim=0).sum(dim=0)

        return stacked_outputs

        
    def forward(self, x):
        self.set_to_train(False)

        outputs = [model(x) for model in self.ensemble]
        weighted_preds = [output * pred * set_weight for output, pred, set_weight in zip(outputs, self.weights, self.set_weights)]

        stacked_outputs = torch.stack(weighted_preds, dim=1).sum(dim=0)

        return stacked_outputs

            
    def train_ensemble(self, optimizer, epoch, scheduler=None):
        '''
        Trains the ensemble for an epoch and optimizes it.

        model: The model to train. Should already be in correct device.
        device: 'cuda' or 'cpu'.
        train_loader: dataloader for training samples.
        optimizer: optimizer to use for model parameter updates.
        epoch: Current epoch/generation in training.

        returns train_loss, train_acc: training loss and accuracy
        '''
        # Empty list to store losses 
        losses = []
        correct, total = 0, 0    
        
        # Iterate over entire training samples in batches
        for batch_sample in tqdm(self.train_loader):
            data, target = batch_sample
            
            # Push data/label to correct device
            data, target = data.to(self.device), target.to(self.device)

            # Reset optimizer gradients. Avoids grad accumulation .
            optimizer.zero_grad()

            output = self._custom_forward(data, training=True)
            
            # target = target.to(torch.float64)

            # Compute loss based on criterion
            loss = self.criterion(output,target)

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
    


    def val_ensemble(self, epoch, final=False, val_loader=None):
        '''
        Tests the model.

        model: The model to train. Should already be in correct device.
        epoch: Current epoch/generation in testing

        returns val_loss, val_acc: validation loss and accuracy
        '''
        losses = []
        correct, total = 0, 0

        if not final and val_loader is None:
            val_loader = self.val_loader        
        
        # Set torch.no_grad() to disable gradient computation and backpropagation
        with torch.no_grad():
            for  sample in tqdm(self.val_loader):
                data, target = sample
                data, target = data.to(self.device), target.to(self.device)
                
            
                if final:
                    output = self.forward(data)
                else:
                    output = self._custom_forward(data, training=False)
                
                # Compute loss based on same criterion as training 
                loss = self.criterion(output,target)
                
                # Append loss to overall test loss
                losses.append(loss.item())
                
                # Get predicted index by selecting maximum log-probability
                pred = output.argmax(dim=1, keepdim=True)
                total += len(target)
                # ======================================================================
                # Count correct predictions overall 
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss = float(np.mean(losses))
        val_acc = (correct / total) * 100.

        divider = "="*70
        test_type = f'{divider}Validation at epoch {epoch}{divider}' \
            if not final else f'{divider}Final Validation{divider}'
        
        print(test_type)
        print(f'\nAverage loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({val_acc:.2f}%)\n')
        print(f'{divider*2}')
        return val_loss, val_acc


def optimizer_type(args, model):
    if args.optimizer == 'Adam' or args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW' or args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD' or args.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer not supported')

def select_model_and_start(args, train_loader, val_loader, num_classes):
    # Selects the models to use and instantiates the ensemble model.
    hands_cnn = Hands_VGG16(args, num_classes=num_classes)
    hands_cnn.load_state_dict(torch.load(args.hands_cnn_path))

    face_cnn = Face_CNN(args, num_classes=num_classes)
    face_cnn.load_state_dict(torch.load(args.face_cnn_path))

    raw_cnn = Raw_CNN(args, num_classes=num_classes)
    raw_cnn.load_state_dict(torch.load(args.raw_cnn_path))
    raw_cnn.eval()

    cnns = [Hands_Inference_Wrapper(hands_cnn, detector_path=args.hands_detector_path), Face_Inference_Wrapper(face_cnn), raw_cnn]

    model = Ensemble(args, cnns, train_loader, val_loader, num_classes=num_classes)
    
    return model

def get_transforms():
    train_transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize((299,299)),
        v2.RandomHorizontalFlip(p=0.4),
        v2.RandomPerspective(distortion_scale=0.1, p=0.25),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.3102, 0.3102, 0.3102], std=[0.3151, 0.3151, 0.3151])
    ])

    test_transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize((299,299)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.3879, 0.3879, 0.3879], std=[0.3001, 0.3001, 0.3001])
    ])

    return train_transform, test_transform

def run_main(args):

    train_transform, test_transform = get_transforms()

    
    train_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/train', cam2_path=f'{args.data_dir}/Camera 2/train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = V2Dataset(cam1_path=f'{args.data_dir}/Camera 1/test', cam2_path=f'{args.data_dir}/Camera 2/test', transform=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    num_classes = len(train_dataset.classes)

    model = select_model_and_start(args, train_loader, val_loader, num_classes)

    model.to(args.device)
    print(model)

    optimizer = optimizer_type(args, model) 

    save_dir =  os.path.join(args.save_dir, args.optimizer)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Creating save directory at '{save_dir}'")

    if args.train:
        # begin genetic algorithm
        if args.resume_path is not None:
            model.load_last_learnable_param(args.resume_path)
            epoch_start = int(args.resume_path.split('/')[-1].split('_')[0].split('epoch')[-1])
            print(f'Resuming from {args.resume_path} at epoch {epoch_start}')

        else:
            epoch_start = 0

        # initialize the best loss to infinity so that the first loss is always better
        best_loss = np.inf
        # train for a given set of epochs, run validation every five, and save the model if the loss is the best so far
        for epoch in range(epoch_start, args.epochs + 1):
            loss, train_acc = model.train_ensemble(optimizer, epoch)
            val_loss, val_acc = model.val_ensemble(epoch)

            with open(f'{save_dir}/losses_and_acc.txt', 'a') as f:
                f.write(f"==================Metrics at epoch {epoch}==================\n \
            Train--> Loss: {loss}, Accuracy: {train_acc:.2f} \n \
            Val--> Loss: {val_loss}, Accuracy: {val_acc:.2f}\n\n")

            if loss < best_loss and epoch % args.save_period == 0:
                best_loss = loss

                now = datetime.now()
                time_now = now.strftime('%m-%d_%H:%M:%S')


                torch.save(model.state_dict(), f'{save_dir}/epoch{epoch}_{time_now}_{val_acc:.0f}acc.pt')
                print(f'Saved model at epoch {epoch}')


        model.save_last_learnable_param(f'{save_dir}/final_ensemble_weights_{datetime.now().strftime("%m-%d_%Hhrs")}.pt')
    else:
        model.load_state_dict(torch.load(args.resume_final_path))




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--distributed', type=bool, default=False)

    args.add_argument('--train', action = 'store_false')
    args.add_argument('--resume_path', type=str, default=None)
    args.add_argument('--final_ensemble_path',  type=str, default=None)

    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--freeze', type=bool, default=True)
    args.add_argument('--dropout', type=float, default=0.5)

    args.add_argument('--epochs', type=int, default=60)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--optimizer', type=str, default='sgd')
    args.add_argument('--weight_decay', type=float, default=1e-6)
    args.add_argument('--scheduler', action='store_true')

    args.add_argument('--save_dir', type=str, default='ensemble_learnable')
    

    args.add_argument('--save_period', type=int, default=2)
    args.add_argument('--device', type=str, default='cuda')

    args.add_argument('--data_dir', type=str, default='data/v2_cam1_cam2_split_by_driver')
    args.add_argument('--raw_model_dir', type=str, default='cnn/hands_models')
    args.add_argument('--face_model_dir', type=str, default='cnn/hands_models')
    args.add_argument('--hands_model_dir', type=str, default='cnn/hands_models')

    args.add_argument('--face_detector_path', type=str, default='/home/ron/Classes/CV-Systems/cybertruck/detection/face_detection/weights/yolov8n-face.pt')
    args.add_argument('--hands_detector_path', type=str, default='/home/ron/Classes/CV-Systems/cybertruck/detection/hands_detection/runs/detect/best/weights/best.pt')

    args.add_argument('--raw_cnn_path', type=str, default='/home/ron/Classes/CV-Systems/cybertruck/cnn/raw_models/raw/SGD/epoch20_11-27_16:15:10_76acc.pt')
    args.add_argument('--face_cnn_path', type=str, default='/home/ron/Classes/CV-Systems/cybertruck/cnn/face_models/face/SGD/epoch10_11-28_10:50:06_66acc.pt')
    args.add_argument('--hands_cnn_path', type=str, default='/home/ron/Classes/CV-Systems/cybertruck/cnn/hands_models/vgg/epoch60_11-16_03:44:44.pt')


    args = args.parse_args()

    run_main(args)