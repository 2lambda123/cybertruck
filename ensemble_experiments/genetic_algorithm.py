import torch
from torch import nn
import numpy as np

from cnn.train_val import train, val

def initialize_population(pop_size, num_models):
    return np.random.rand(pop_size, num_models)

def calculate_fitness(weights, ensemble, train_loader, val_loader, criterion, optimizer, device, generation):

    ensemble_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    for model, weight in zip(ensemble, ensemble_weights):
        for param in model.parameters():
            param.data = param.data * weight

    train_loss, _ = train(ensemble, None, device, train_loader, optimizer=optimizer, criterion=criterion, epoch=generation)
    val_loss, _ = val(ensemble, None, device, val_loader, criterion, epoch=0)

    fitness = 0.8 * (1 / (1 + train_loss)) + 0.2 * (1 / (1 + val_loss))
    return fitness

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(child, mutation_rate):
    mutation_mask = np.random.rand(len(child)) < mutation_rate
    child[mutation_mask] = np.random.rand(np.sum(mutation_mask))
    return child

def select_parents(population, fitness_scores):
    tournament_size = 3
    selected_parents = []

    for _ in range(len(population) // 2):
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        selected_parents.extend([population[i] for i in tournament_indices[np.argmax(tournament_fitness)]])

    return selected_parents

def genetic_algorithm(ensemble, num_models, train_loader, val_loader, criterion, optimizer, args):
    pop_size = 10
    mutation_rate = 0.1
    num_generations = 10

    # num_models = num_models
    population = initialize_population(pop_size, num_models)

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(weights, ensemble, train_loader, val_loader, criterion, optimizer, args.device, generation) for weights in population]

        parents = select_parents(population, fitness_scores)

        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            offspring.extend([child1, child2])

        population = np.array(offspring)

    best_weights = population[np.argmax(fitness_scores)]
    return best_weights
