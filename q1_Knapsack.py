from random import randint, random, choices
import sys
from matplotlib import pyplot as plt
import numpy as np
import time
import heapq

from GAfuncs import evolve_population, plot_scores #evolve_population, plot_scores

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

def objective(individual):
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

alpha = -1
global capacity
global items

def Knapsack(file_capacity, file_items, seed):
    global capacity
    capacity = file_capacity
    global items
    items = file_items

    #hyperparameters
    populationSize = 300
    numEpochs = 100
    global alpha
    alpha = 5 # a -1 means invalid solution = 0
    mutation_rate = 0.5 #0.2
    elitism =  0.05
    geneweights = []
    for item in items:
        geneweights.append(item['value']/item['weight'])


    # population = [[randint(0, 1) for x in range(numitems)] for _ in range(populationSize)]
    population = [[0 for x in range(len(items))] for _ in range(populationSize)]

    best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = evolve_population(population, numEpochs, objective, geneweights, mutation_rate, populationSize, elitism, seed)

    return best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores

if __name__ == '__main__':
    if len(sys.argv) == 2:
        random_seed = 100
        capacity, items = process_file(sys.argv[1])
        best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = Knapsack(capacity, items, random_seed)

        print("Best score = " + str(best_feasible_score))
        print("Best solution = " + str(best_feasible_solution))
        plot_scores(best_feasible_scores, ave_scores)

    else:
        print('You need to input the path to the knapsack file to run')

#TODO:
# - check if the accuracy being 1513 instead of 1514 matters
#   - if it does, look at advanced knapsack techniques specified in the slide
# - cut down on epoch number
# - updated README