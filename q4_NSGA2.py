from functools import partial
import math
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
import time
import heapq
import operator
from deap import gp, creator, base, tools, algorithms
from scipy.stats import mstats

from sklearn.model_selection import train_test_split

from GAfuncs import evolve_population, plot_scores  # evolve_population, plot_scores

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
        return sum_value, True 
    

def WrapperGA():
    #hyperparameters
    populationSize = 100
    numEpochs = 40
    mutation_rate = 0.2
    elitism =  0.05
    geneweights = []
    for i in range(len(featurenames)):
        geneweights.append(1)

    # population = [[randint(0, 1) for x in range(numitems)] for _ in range(populationSize)]
    population = [[0 for x in range(len(featurenames))] for _ in range(populationSize)]

    best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = evolve_population(population, numEpochs, wrapper_objective, geneweights, mutation_rate, populationSize, elitism)

    print("Best score = " + str(best_feasible_score))
    print("Best solution = " + str(best_feasible_solution))
    plot_scores(best_feasible_scores, ave_scores)

def NSGA2():
    #TODO
    x = 1

if __name__ == "__main__":
    if len(sys.argv) == 1:
        NSGA2()
    else:
        print("Too many CMD arguments")
