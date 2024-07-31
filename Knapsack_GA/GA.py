from random import randint, random
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
    for x in individual:
        if x == 1:
            sum_weights += items[x]['weight']
            sum_value += items[x]['value']
    
    if sum_weights > capacity:
        return sum_value - (sum_weights - capacity)*alpha, False
    else:
        return sum_value, True

def plot_scores(scores):
    # Create a range of epochs
    epoch_range = range(1, len(scores) + 1)

    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, scores)

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Best Scores')
    plt.title('Best Scores over Epochs')

    # Show the plot
    plt.show()

def GA(capacity, items):
    numitems = len(items)

    #hyperparameters
    populationSize = 100
    numEpochs = 200
    alpha = 2
    mutation_rate = 0.2
    elitism =  0.05

    best_feasible_score = -1
    best_feasible_solution = -1
    num_elitismed = int(populationSize * elitism)
    best_feasible_scores = []

    population = [[randint(0, 1) for x in range(numitems)] for _ in range(populationSize)]
    for epoch in range(numEpochs):
        newpopulation = []
        scores = []
        sum_scores = 0
        for individual in population:
            score, isfeasible = objective(individual, items, capacity, alpha)
            if score > best_feasible_score and isfeasible:
                best_feasible_score = score
                best_feasible_solution = individual
            scores.append((score, individual))
            sum_scores += score
        
        best_feasible_scores.append(best_feasible_score)

        #Elitism
        elitist_individuals = heapq.nlargest(num_elitismed, scores, key=lambda t: t[0])
        elitist_individuals = [x[1] for x in elitist_individuals]
        newpopulation.extend(elitist_individuals)

        #Rioulette selection
        cumulative_probabilities = []
        cumulative_sum = 0.0
        for score, individual in scores:
            relative_fitness = score / sum_scores
            cumulative_sum += relative_fitness
            cumulative_probabilities.append(cumulative_sum)

        while len(newpopulation) < populationSize:
            # Rioulette selection based on cumulative probabilities
            r1 = random()
            r2 = random()
            r1done, r2done = False, False
            indiv1, indiv2 = None, None 
            for i, individual in enumerate(population):
                if not r1done and r1 <= cumulative_probabilities[i]:
                    indiv1 = individual
                    r1done = True
                if not r2done and r2 <= cumulative_probabilities[i]:
                    indiv2 = individual
                    r2done = True
                if r1done and r2done:
                    break
            
            #Crossover
            crossoverPoint = randint(1, numitems - 1)
            for x in range(crossoverPoint):
                temp = indiv1[x]
                indiv1[x] = indiv2[x]
                indiv2[x] = temp
        
            #Mutation
            if random() < mutation_rate: #mutate child 1
                flipindex = randint(0, numitems - 1)
                indiv1[flipindex] = abs(indiv1[flipindex] - 1) #flip from 0 to 1 and vice versa
            if random() < mutation_rate: #mutate child 2
                flipindex = randint(0, numitems - 1)
                indiv2[flipindex] = abs(indiv2[flipindex] - 1) #flip from 0 to 1 and vice versa

            newpopulation.append(indiv1)
            newpopulation.append(indiv2)
        
        population = newpopulation

    print("Best score = " + str(best_feasible_score))
    print("Best solution = " + str(best_feasible_solution))
    plot_scores(best_feasible_scores)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        capacity, items = process_file(sys.argv[1])
        GA(capacity, items)
    else:
        print('You need to input the path to the knapsack file to run')