import heapq
from random import choices, randint, random
import matplotlib.pyplot as plt
import sys

def plot_scores(scores, ave_scores):
    # Create a range of epochs
    epoch_range = range(1, len(scores) + 1)

    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, scores, label='Best scores')
    plt.plot(epoch_range, ave_scores, label='Average scores')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.title('Best Scores over Epochs')
    plt.legend()

    # Show the plot
    plt.show()

def make_new_population(population, numgenes, num_elitismed, scores, sum_scores, populationSize, mutation_rate, geneweights):
    newpopulation = []

    #Elitism
    elitist_individuals = heapq.nlargest(num_elitismed, scores, key=lambda t: t[0])
    elitist_individuals = [x[1] for x in elitist_individuals]
    newpopulation.extend(elitist_individuals)

    #Get biases for rioulette selection
    fitness_values = []
    for score, individual in scores:
        adjusted_fitness = score
        adjusted_score_sum = sum_scores #- min_score * len(population) + (0.000000001) * len(population)
        relative_fitness = adjusted_fitness / adjusted_score_sum
        fitness_values.append(relative_fitness)

    while len(newpopulation) < populationSize:
        # Rioulette selection based on cumulative probabilities
        parent1 = choices(population, weights=fitness_values, k=1)[0]
        parent2 = choices(population, weights=fitness_values, k=1)[0]
        
        
        #Crossover
        crossoverPoint = randint(0, numgenes - 1)
        child1 = parent1[0:crossoverPoint] + parent2[crossoverPoint:]
        child2 = parent2[0:crossoverPoint] + parent1[crossoverPoint:]

        #Mutation
        if random() < mutation_rate: #mutate child 1
            flipindex = choices([x for x in range(0, numgenes)], weights=geneweights, k=1)[0]
            child1[flipindex] = abs(child1[flipindex] - 1) #flip from 0 to 1 and vice versa
        if random() < mutation_rate: #mutate child 2
            # flipindex = randint(0, numitems - 1)
            flipindex = choices([x for x in range(0, numgenes)], weights=geneweights, k=1)[0]
            child2[flipindex] = abs(child2[flipindex] - 1) #flip from 0 to 1 and vice versa

        newpopulation.append(child1)
        newpopulation.append(child2)

    return newpopulation

def evolve_population(population, numEpochs, alpha, objective, geneweights, mutation_rate, populationSize, elitism):
    numgenes = len(population[0])
    best_feasible_score = -sys.maxsize
    best_feasible_solution = -1
    num_elitismed = int(populationSize * elitism)
    best_feasible_scores = []
    ave_scores = []

    for epoch in range(numEpochs):
        scores = []
        sum_scores = 0
        min_score = sys.maxsize
        best_score_this_epoch = 0
        best_indiv_this_epoch = None

        #Calculate scores
        for individual in population:
            score, isfeasible = objective(individual, alpha)
            if score > best_score_this_epoch and isfeasible:
                best_score_this_epoch = score
                best_indiv_this_epoch = individual
            if score < min_score:
                min_score = score
            scores.append((score, individual))
            sum_scores += score
        
        if best_score_this_epoch > best_feasible_score:
            best_feasible_score = best_score_this_epoch
            best_feasible_solution = best_indiv_this_epoch

        best_feasible_scores.append(best_score_this_epoch)
        ave_scores.append(sum_scores/len(population))
        
        population = make_new_population(population, numgenes, num_elitismed, scores, sum_scores, populationSize, mutation_rate, geneweights)
    
    return best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores