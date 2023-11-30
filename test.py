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
import random
import numpy as np



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
        self.weights = nn.Parameter(torch.ones(num_models) / num_models, requires_grad=True)
        # self.softmax = nn.Softmax(dim=1)
        # self.classifier = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(num_classes * num_models, num_classes)
        #     )

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

    def _custom_forward(self, x, training=True):        
        with torch.set_grad_enabled(training):
            # outputs = [self.softmax(model(x)) for model in self.models]
            outputs = [model(x) for model in self.ensemble]
            weighted_preds = [output * pred for output, pred in zip(outputs, self.weights)]
            stacked_outputs = torch.stack(weighted_preds, dim=0).sum(dim=0)
            return stacked_outputs
            # return self.classifier(stacked_outputs)
        
    def forward(self, x):
        with torch.set_grad_enabled(False):
            # outputs = [self.softmax(model(x)) for model in self.models]
            outputs = [model(x) for model in self.ensemble]
            weighted_preds = [output * pred for output, pred in zip(outputs, self.weights)]
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
    


    def val_ensemble(self, epoch):
        '''
        Tests the model.

        model: The model to train. Should already be in correct device.
        epoch: Current epoch/generation in testing

        returns val_loss, val_acc: validation loss and accuracy
        '''
        losses = []
        correct, total = 0, 0
        
        # Set torch.no_grad() to disable gradient computation and backpropagation
        with torch.no_grad():
            for  sample in tqdm(self.val_loader):
                data, target = sample
                data, target = data.to(self.device), target.to(self.device)
                
            
                output = self._custom_forward(data, training=False)
                # output = self.ensemble(data, custom_forward=True, training=False)
                
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

        print(f'==========================Validation at epoch {epoch}==========================')
        print(f'\nAverage loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({val_acc:.2f}%)\n')
        print(f'===============================================================================')
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
    # cnns = [Hands_Inference_Wrapper(hands_cnn, detector_path=args.hands_detector_path)]

    model = Ensemble(args, cnns, train_loader, val_loader, num_classes=num_classes)
    
    return model

def get_transforms():
    train_transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize((299,299)),
        v2.RandomHorizontalFlip(p=0.4),
        v2.RandomPerspective(distortion_scale=0.1, p=0.25),
        # v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
            loss, _ = model.train_ensemble(optimizer, epoch)
            model.val_ensemble(epoch)

            if loss < best_loss and epoch % args.save_period == 0:
                best_loss = loss

                now = datetime.now()
                time_now = now.strftime('%m-%d_%H:%M:%S')

                save_dir =  os.path.join('ALT_ensemble_weights', args.optimizer)

                os.makedirs(save_dir, exist_ok=True)

                torch.save(model.state_dict(), f'{save_dir}/epoch{epoch}_{time_now}.pt')
                print(f'Saved model at epoch {epoch}')
            

        save_dir = 'ensemble_weights'
        os.makedirs(save_dir, exist_ok=True)
        # torch.save(model.state_dict(), f'{save_dir}/final_ensemble_weights_{datetime.now().strftime("%m-%d_%Hhrs")}.pt')
        model.save_last_learnable_param(f'{save_dir}/final_ensemble_weights_{datetime.now().strftime("%m-%d_%Hhrs")}.pt')
    else:
        model.load_state_dict(torch.load(args.resume_final_path))




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--distributed', type=bool, default=False)

    args.add_argument('--train', type=bool, default=True)
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
    class GeneticAlgorithm:
        def __init__(self, population_size, num_weights):
            self.population_size = population_size
            self.num_weights = num_weights

        def initialize_population(self):
            population = []
            for _ in range(self.population_size):
                weights = [random.uniform(0, 1) for _ in range(self.num_weights)]
                population.append(weights)
            return population

        def evaluate_fitness(self, population):
            fitness_scores = []
            for weights in population:
                # Set the weights in the ensemble model
                model.set_weights(weights)

                # Evaluate the ensemble model on the validation set
                _, accuracy = model.val_ensemble(epoch)

                # Higher accuracy is better, so use negative accuracy as fitness score
                fitness_scores.append(-accuracy)
            return fitness_scores

        def select_parents(self, population, fitness_scores, num_parents):
            parents = []
            for _ in range(num_parents):
                # Select two random individuals from the population
                idx1, idx2 = random.sample(range(len(population)), 2)

                # Choose the individual with the better fitness score as a parent
                if fitness_scores[idx1] < fitness_scores[idx2]:
                    parents.append(population[idx1])
                else:
                    parents.append(population[idx2])
            return parents

        def crossover(self, parents, num_offspring):
            offspring = []
            for _ in range(num_offspring):
                # Select two random parents
                parent1, parent2 = random.sample(parents, 2)

                # Perform uniform crossover to create a new offspring
                weights = []
                for i in range(self.num_weights):
                    if random.random() < 0.5:
                        weights.append(parent1[i])
                    else:
                        weights.append(parent2[i])
                offspring.append(weights)
            return offspring

        def mutate(self, population, mutation_rate):
            for i in range(len(population)):
                for j in range(self.num_weights):
                    # Apply mutation with a certain probability
                    if random.random() < mutation_rate:
                        # Generate a random weight between 0 and 1
                        population[i][j] = random.uniform(0, 1)
            return population

        def run(self, num_generations, mutation_rate):
            population = self.initialize_population()

            for generation in range(num_generations):
                fitness_scores = self.evaluate_fitness(population)
                parents = self.select_parents(population, fitness_scores, num_parents=2)
                offspring = self.crossover(parents, num_offspring=self.population_size - len(parents))
                population = parents + offspring
                population = self.mutate(population, mutation_rate)

            # Select the best individual from the final population
            best_weights = max(population, key=lambda weights: self.evaluate_fitness([weights])[0])
            return best_weights

    # Create an instance of the genetic algorithm
    ga = GeneticAlgorithm(population_size=100, num_weights=3)

    # Run the genetic algorithm for 100 generations with a mutation rate of 0.1
    best_weights = ga.run(num_generations=100, mutation_rate=0.1)

    # Set the best weights in the ensemble model
    model.set_weights(best_weights)
