from random import randint, random, choices
import sys
from matplotlib import pyplot as plt
import numpy as np
import time
import heapq

def process_file(filename):
    numitems = -1
    capacity = -1
    items = []
    with open(filename, 'r') as file:
        firstline = True
        for line in file:
            words = line.split()
            if firstline:
                numitems = int(words[0])
                capacity = int(words[1])
                firstline = False       
            else:     
                items.append( {'value': int(words[0]), 'weight': int(words[1])})

    if not int(numitems) == len(items):
        print("ERROR ERROR NUMBER OF ITEMS SPECIFIED DOESNT MATCH NUMBER OF ITEMS READ")
    return capacity, items    

def objective(individual, items, capacity, alpha):
    sum_weights = 0
    sum_value = 0
    for index, x in enumerate(individual):
        if x == 1:
            sum_weights += items[index]['weight']
            sum_value += items[index]['value']
    
    if sum_weights > capacity: #invalid
        if alpha == -1:
            return 0, False
        else:
            return max(0, sum_value - (sum_weights - capacity)*alpha), False
    else:
        if sum_value == 0:
            return 1, True
        return sum_value, True #TODO remember you wrote 1+ here

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

def GA(capacity, items):

    numitems = len(items)

    #hyperparameters
    populationSize = 100
    numEpochs = 500
    # alpha = 0.5
    alpha = 3 #this means invalid solution = 0
    mutation_rate = 0.2
    elitism =  0.05

    best_feasible_score = -sys.maxsize
    best_feasible_solution = -1
    num_elitismed = int(populationSize * elitism)
    best_feasible_scores = []
    ave_scores = []

    # population = [[randint(0, 1) for x in range(numitems)] for _ in range(populationSize)]
    population = [[0 for x in range(numitems)] for _ in range(populationSize)]

    for epoch in range(numEpochs):
        newpopulation = []
        scores = []
        sum_scores = 0
        min_score = sys.maxsize
        best_score_this_epoch = -sys.maxsize
        best_indiv_this_epoch = None
        for individual in population:
            score, isfeasible = objective(individual, items, capacity, alpha)
            if score > best_score_this_epoch and isfeasible:
                best_score_this_epoch = score
                best_indiv_this_epoch = individual
            if score < min_score:
                min_score = score
            scores.append((score, individual))
            sum_scores += score
            # print(score)
        
        if best_score_this_epoch > best_feasible_score:
            best_feasible_score = best_score_this_epoch
            best_feasible_solution = best_indiv_this_epoch
        # print(sum_scores)
        # print(population)
        # print(scores)
        # print("sumscores = " + str(sum_scores))
        # print("popsize = " + str(len(population)))

        best_feasible_scores.append(best_score_this_epoch)
        ave_scores.append(sum_scores/len(population))
        print(sum_scores/len(population))
        print(best_score_this_epoch)

        #Elitism
        elitist_individuals = heapq.nlargest(num_elitismed, scores, key=lambda t: t[0])
        elitist_individuals = [x[1] for x in elitist_individuals]
        newpopulation.extend(elitist_individuals)


        #Rioulette selection
        # if sum_scores - min_score * len(population) == 0:
        #     print(scores)
        
        fitness_values = []

        for score, individual in scores:
            adjusted_fitness = score
            adjusted_score_sum = sum_scores #- min_score * len(population) + (0.000000001) * len(population)
            relative_fitness = adjusted_fitness / adjusted_score_sum
            fitness_values.append(relative_fitness)
        # print(sum_scores)

        while len(newpopulation) < populationSize:
            # Rioulette selection based on cumulative probabilities
            parent1 = choices(population, weights=fitness_values, k=1)[0]
            parent2 = choices(population, weights=fitness_values, k=1)[0]
            
            
            #Crossover
            # print(str(indiv1) + " " + str(indiv2))
           
            crossoverPoint = randint(0, numitems - 1)
            child1 = parent1[0:crossoverPoint] + parent2[crossoverPoint:]
            child2 = parent2[0:crossoverPoint] + parent1[crossoverPoint:]
            # print(str(indiv1) + " " + str(indiv2))

            #Mutation
            if random() < mutation_rate: #mutate child 1
                flipindex = randint(0, numitems - 1)
                child1[flipindex] = abs(child1[flipindex] - 1) #flip from 0 to 1 and vice versa
            if random() < mutation_rate: #mutate child 2
                flipindex = randint(0, numitems - 1)
                child2[flipindex] = abs(child2[flipindex] - 1) #flip from 0 to 1 and vice versa

            newpopulation.append(child1)
            newpopulation.append(child2)
            # print(str(indiv1) + " " + str(indiv2))
        
        population = newpopulation

    print("Best score = " + str(best_feasible_score))
    print("Best solution = " + str(best_feasible_solution))
    plot_scores(best_feasible_scores, ave_scores)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        capacity, items = process_file(sys.argv[1])
        GA(capacity, items)
    else:
        print('You need to input the path to the knapsack file to run')

#TODO:
# - by the end of training all of the solutions are identical. Also sometimes early in the training
# - because so many are identical, you can get a score sum of 0 (as all individuals are [0...0]) and get divideby 0 error
# - sum of scores being negative breaks rioulette wheel and training I think
# - check that the index 1 fitness = index i population = index i everything so that fitness accuracy maps to indivudal choice