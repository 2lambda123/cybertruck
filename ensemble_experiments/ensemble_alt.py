
import os
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

        self.optimal_weights = None

    def __len__(self):
        return self.num_models
    
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

        self.optimal_weights = None #TODO load optimal weights from file

        if self.optimal_weights is None:
            raise RuntimeError("Ensemble weights not set. Run genetic_alg() to set the weights.")

        weighted_predictions = torch.stack([model(x) * weight for model, weight in zip(self.ensemble, self.optimal_weights)])
        final_prediction = torch.sum(weighted_predictions, dim=0)

        return final_prediction
    

    def _train_ensemble(self, optimizer, epoch, scheduler=None):
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
            # optimizer.zero_grad()

            output = self._custom_forward(data, training=True)
            
            # target = target.to(torch.float64)

            # Compute loss based on criterion
            loss = self.criterion(output,target)

            # Computes gradient based on final loss
            # loss.backward()
            
            # Store loss
            losses.append(loss.item())
            
            # Optimize model parameters based on learning rate and gradient 
            # optimizer.step()

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
            for  sample in tqdm(val_loader):
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


    def _initialize_population(self, pop_size, num_models):
        '''
        Initializes the population for the genetic algorithm.

        pop_size: size of the population
        num_models: number of models in the ensemble

        returns population: random weight initialization for the ensemble
        '''
        return torch.rand(pop_size, num_models)


    def _calculate_fitness(self, weights, optimizer, generation):
        '''
        Calculates the fitness of a given set of weights.

        weights: weights to use for the ensemble
        optimizer: optimizer to use for model parameter updates.
        generation: current generation/epoch

        returns fitness: fitness score of the given weights
        '''
        self.ensemble_weights = torch.tensor(weights, dtype=torch.float32, device=self.args.device)
        
        # Set the weights of the ensemble to the current weights
        for model, weight in zip(self.ensemble, self.ensemble_weights):
            for param in model.parameters():
                # TODO need to experiment if multiplying or replacing the weights yields best results
                param.data = param.data * weight
                # param.data = weight

        # train and validate the ensemble with the current weights
        train_loss, _ = self._train_ensemble(optimizer=optimizer, epoch=generation)
        val_loss, _ = self.val_ensemble(epoch=generation)
        

        # fitness is a weighted average of the training and validation loss.
        fitness = 0.8 * (1 / (1 + train_loss)) + 0.2 * (1 / (1 + val_loss))
        # fitness = 1 / (1 + val_loss)
        return fitness

    def _crossover(self, parent1, parent2):
        '''
        Performs a single point crossover between two parents. Analogous to biological passing of genes.

        parent1: first parent
        parent2: second parent

        returns child1, child2: two children, product of the crossover between the two parents
        '''
        upper_bound = len(parent1) - 1
        
        # randomly select a crossover point
        crossover_point = torch.randint(1, upper_bound, (1,)).item()

        child1 = torch.concatenate((torch.stack(parent1[:crossover_point]), torch.stack(parent2[crossover_point:])))
        child2 = torch.concatenate((torch.stack(parent2[:crossover_point]), torch.stack(parent1[crossover_point:])))

        return child1, child2

    def _mutate(self, child, mutation_rate):
        '''
        Mutates a child by randomly changing some of its weights.

        child: child to mutate
        mutation_rate: probability of mutation

        returns child: mutated child
        '''
        mutation_mask = np.random.rand(len(child)) < mutation_rate
        
        mutations = np.random.rand(np.sum(mutation_mask))
        if len(mutations) > 0:
            child[mutation_mask] = torch.tensor(mutations).unsqueeze(1).to(torch.float32)

        return child

    def _select_parents(self, population, fitness_scores):
        ''' 
        Selects parents for crossover using tournament selection.

        population: population of weights
        fitness_scores: fitness scores of each weight

        returns selected_parents
        '''
        tournament_size = 3
        selected_parents = []
        parent = []

        for _ in range(len(population) // 2):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]

            sorted_fitness_scores = np.argsort(tournament_fitness)[::-1]
            for fitness_score in sorted_fitness_scores:
                parent.append(tournament_indices[fitness_score])

            selected_parents.append([population[i] for i in parent])

        return selected_parents

    def _genetic_algorithm(self, optimizer, save_dir, resume=False):
        '''
        Runs the genetic algorithm to optimize ensemble weights

        optimizer: optimizer to use for model parameter updates.
        '''
        # Hyperparameters for genetic algorithm
        mutation_rate = 0.1
        num_generations = 5
        pop_size = 10
        
        population = self._initialize_population(pop_size, self.num_models)


        for generation in range(num_generations):
            fitness_scores = [self._calculate_fitness(weights, optimizer, generation) for weights in population]

            parents = self._select_parents(population, fitness_scores)

            offspring = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1, mutation_rate)
                child2 = self._mutate(child2, mutation_rate)
                offspring.extend([child1, child2])

            # Replace the population with the offspring and repeat the process for the next generation
            population = torch.cat(offspring, dim = 0)
            offspring.clear()

            with open(f'{save_dir}/weights_values_during_gen.txt', 'a') as f:
                f.write(f"Generation {generation}:\n Best Weights: {population[np.argmax(fitness_scores)]}\n Best Fitness Score: {np.max(fitness_scores)}\n\n")
                
        # Set the weights of the ensemble to the best weights
        self.ensemble_weights = population[np.argmax(fitness_scores)]



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

    optimizer = optimizer_type(args, model) 

    save_dir = os.path.join(args.save_dir, args.optimizer)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Creating save directory at '{save_dir}'")

    if args.train:
        model._genetic_algorithm(optimizer, save_dir=save_dir)
    else:
        print('Running Validation with GA Weights')
        model.val_ensemble(epoch=None, final=True, val_loader=val_loader)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--distributed', type=bool, default=False)

    args.add_argument('--train', action = 'store_false')
    args.add_argument('--resume_path', type=str, default=None)
    args.add_argument('--final_ensemble_path',  type=str, default=None)

    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--freeze', type=bool, default=True)
    args.add_argument('--dropout', type=float, default=0.5)

    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--optimizer', type=str, default='sgd')
    args.add_argument('--weight_decay', type=float, default=0.0)
    args.add_argument('--scheduler', action='store_true')

    args.add_argument('--save_dir', type=str, defaul='ensemble_alt')
    

    args.add_argument('--save_period', type=int, default=5)
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