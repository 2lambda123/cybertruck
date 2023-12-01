import os
from typing import Any
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

class GeneticAlgorithm:
    def __init__(self, model, population_size, num_weights, save_dir):
        self.population_size = population_size
        self.num_weights = num_weights
        self.model = model
        self.save_dir = save_dir

        self.optimal_weights = None


    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            weights = [random.uniform(0, 1) for _ in range(self.num_weights)]
            population.append(weights)
        return population

    def evaluate_fitness(self, population, epoch):
        fitness_scores = []
        for weights in population:
            # Set the weights in the ensemble model
            self.model.set_weights(weights)

            # Evaluate the ensemble model on the validation set
            _, accuracy = self.model.val_ensemble(epoch)

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
            fitness_scores = self.evaluate_fitness(population, epoch=generation)
            parents = self.select_parents(population, fitness_scores, num_parents=2)
            offspring = self.crossover(parents, num_offspring=self.population_size - len(parents))
            population = parents + offspring
            population = self.mutate(population, mutation_rate)

            with open(f'{self.save_dir}/weights_values_during_gen.txt', 'a') as f:
                f.write(f"Generation {generation}:\n Best Weights: {population[np.argmax(fitness_scores)]}\n\n")

        # Select the best individual from the final population
        best_weights = max(population, key=lambda weights: self.evaluate_fitness([weights], 0)[0])
        return best_weights

class Ensemble(nn.Module):
    '''
    Ensemble model that takes in a list of models and optimizes their weights using a genetic algorithm

    args: hyperparameters from command line
    models: The models to train, passed as a list. Should already be in correct device.
    train_loader: dataloader for training samples.
    val_loader: dataloader for validation samples.
    num_classes: 10 classes for dataset.
    '''
    def __init__(self, args, models, train_loader, val_loader, num_classes=10):
        super(Ensemble, self).__init__()
        self.args = args
        self.num_models = len(models)
        self.device = args.device

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()

        self.ensemble = nn.ModuleList([model for model in models])

        self.classifier = nn.Linear(num_classes, num_classes)
        self.ensemble_weights = None
        self.freeze_all_weights()



    def __len__(self):
        return self.num_models
    
    def set_weights(self, weights):
        self.ensemble_weights = weights
    
    def freeze_all_weights(self):
        for model in self.ensemble:
            for _, param in model.named_parameters():
                if param.requires_grad == True:
                    param.requires_grad = False


    def _custom_forward(self, x, training=True):
        '''
        Method used to forward pass through the ensemble model as it learns with the genetic algorithm.

        x: input data
        training: boolean to indicate whether the model is training or not.

        returns prediction: the prediction given during training/validation in genetic algorithm
        '''

        # sets the gradient calculation to be enabled or disabled based on training.
        # models in ensemble don't need to be trained, so we only need to set this portion to be trainable.
        weighted_predictions = torch.stack([model(x) * weight for model, weight in zip(self.ensemble, self.ensemble_weights)])
        final_prediction = torch.sum(weighted_predictions, dim=0)

        return final_prediction



    #TODO not yet tested. Likely wrong implementation.  Will test once we have a trained ensemble model.
    def forward(self, x):
        '''
        Once the genetic algorithm has finished optimizing the weights, this method is used to forward pass through the ensemble model.

        x: input data

        returns final prediction
        '''
        ga_weights  = torch.tensor([0.1646, 0.1443, 0.8192])
        normalized_weights = ga_weights / torch.sum(ga_weights)
        self.optimal_weights = normalized_weights


        if self.optimal_weights is None:
            raise RuntimeError("Ensemble weights not set. Run genetic_alg() to set the weights.")

        weighted_predictions = torch.stack([model(x) * weight for model, weight in zip(self.ensemble, self.optimal_weights)])
        final_prediction = torch.sum(weighted_predictions, dim=0)

        return final_prediction    


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
            for  sample in tqdm(val_loader):
                data, target = sample
                data, target = data.to(self.device), target.to(self.device)
                
                if final:
                    output = self.forward(data)
                else:
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

    cnns = [Hands_Inference_Wrapper(hands_cnn, detector_path=args.hands_detector_path), Face_Inference_Wrapper(face_cnn, detector_path=args.face_detector_path), raw_cnn]
    # cnns = [Hands_Inference_Wrapper(hands_cnn, detector_path=args.hands_detector_path)]
    # cnns = [raw_cnn]

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

    save_dir = os.path.join(args.save_folder)
    os.makedirs(save_dir, exist_ok=True) 
    print(f"Creating save directory at '{save_dir}'")

    if args.train:
        print('Training Ensemble')
        # Create an instance of the genetic algorithm
        ga = GeneticAlgorithm(model=model, population_size=args.pop_size, num_weights=args.num_weights, save_dir=save_dir)

        # Run the genetic algorithm for 100 generations with a mutation rate of 0.1
        best_weights = ga.run(num_generations=args.num_gens, mutation_rate=0.1)

        # Set the best weights in the ensemble model
        model.set_weights(best_weights)

        
        with open(f'{save_dir}/store_weights_values.txt', 'a') as f:
                    f.write(f"Best Fitness Score: {best_weights}\n\n")
    else:
        print('Running Validation with GA Weights')
        model.val_ensemble(epoch=None, final=True, val_loader=val_loader)


        # print('Running Inference')
        # example = val_dataset[0][0].unsqueeze(0).to(args.device)
        # prediction = model(example)
        # print(f'Prediction: {prediction.argmax(dim=1).item()}')
        # print(f'GT Class: {val_dataset[0][1]}')




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--distributed', type=bool, default=False)

    args.add_argument('--resume_path', type=str, default=None)
    args.add_argument('--final_ensemble_path',  type=str, default=None)


    args.add_argument('--train', action = 'store_false')

    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--freeze', type=bool, default=True)
    args.add_argument('--dropout', type=float, default=0.5)

    args.add_argument('--pop_size', type=int, default=25)
    args.add_argument('--num_gens', type=int, default=12)
    args.add_argument('--num_weights', type=int, default=3)

    args.add_argument('--save_folder', type=str, default='ensemble_weights')
    

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
