import datetime
from functools import partial
import math
import random
import re
import sys
from matplotlib import pyplot as plt
import numpy as np
import time
import heapq
import operator
from deap import gp, creator, base, tools, algorithms
import pandas as pd
from scipy.stats import mstats
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from GAfuncs import evolve_population, plot_scores  # evolve_population, plot_scores

def process_file(folderpath):
    dataNotDat = True
    file_names = os.listdir(folderpath)
    for filename in file_names:
        if ".data" in filename:
            dataNotDat = True
            datafile = os.path.join(folderpath, filename)
        if ".names" in filename:
            namesfile = os.path.join(folderpath, filename)
        if ".dat" in filename and not ".data" in filename:
            dataNotDat = False
            datfile = os.path.join(folderpath, filename)
        if ".doc" in filename:
            docFile = os.path.join(folderpath, filename)

    featureNames = []
    classes = []
    iscontinuous = {}
    discreteOptions = {}

    if dataNotDat:
        with open(namesfile, 'r') as file:
            firstline = True
            for line in file:
                line = line.replace('\r', '')
                if firstline:
                    line = line.replace('.\n', '')
                    classes = line.split(',')
                    firstline = False       
                else:     
                    line = line.split(': ')
                    featureNames.append(line[0])
                    iscontinuous[line[0]] = "continuous" in line[1]
                    if not iscontinuous[line[0]]:
                        options = line[1].split(',')
                        options[-1] = options[-1].replace('\n', '').replace('.', '')
                        discreteOptions[line[0]] = options

        data = []
        with open(datafile, 'r') as file:
            for line in file:
                line = line.replace('\r', '')
                line = line.split(',')
                linedict = {}
                if len(line) != len(featureNames) + 1:
                    print("ERROR")
                for i in range(len(line) - 1):
                    if iscontinuous[featureNames[i]]:
                        linedict[featureNames[i]] = float(line[i])
                    else:
                        linedict[featureNames[i]] = line[i]
                        if not line[i] in discreteOptions[featureNames[i]]:
                            print("ERROR discrete class used but not defined in .names file")
                linedict['class'] = line[-1].replace('\n', '')
                data.append(linedict)
    else:
        with open(docFile, 'r') as file:
            firstline = True
            readingAttributes = False
            readingClasses = False
            readNextLine = False
            for line in file:
                if "ATTRIBUTES" in line:
                    readingAttributes = True
                    continue
                if readingAttributes:
                    if 'NUMBER OF CLASSES' in line:
                        readingAttributes = False
                        readingClasses = True
                        continue
                    line = line.split('\t')
                    if len(line) > 2 or readNextLine or (len(line) == 2 and "MAX.LENGTH" in line[1]):
                        if len(featureNames) > 0 and readNextLine:
                            if not featureNames[-1].endswith(' '):
                                featureNames[-1] = featureNames[-1] + ' '
                            featureNames[-1] = featureNames[-1] + line[1]
                            featureNames[-1] = featureNames[-1].replace('\n', '').strip()
                        else:
                            if len(line[1]) == 0:
                                continue
                            uppercase_words = re.findall(r'\b[A-Z]+\b', line[1])
                            uppercase_words = ' '.join(uppercase_words)
                            if line[1].endswith(' '):
                                uppercase_words = uppercase_words + ' '
                            featureNames.append(uppercase_words)
                        if featureNames[-1].endswith(' ') or featureNames[-1].endswith("ABOUT"):
                            readNextLine = True
                        else:
                            readNextLine = False
                if readingClasses:
                    if '\t' in line:
                        line = line.split('\t')
                        classes = line[2].split(', ')
                        classes[-1] = classes[-1].replace('\n', '')
                        readingClasses = False

        for feature in featureNames:
            iscontinuous[feature] = True

        data = []
        with open(datfile, 'r') as file:
            for line in file:
                line = line.replace('\r', '')
                line = line.split(' ')
                if line[-1] == '\n' or line[-1] == '':
                    del line[-1]
                linedict = {}
                if len(line) != len(featureNames) + 1:
                    print(len(line))
                    print(line)
                    print("ERROR")
                for i in range(len(line) - 1):
                    if iscontinuous[featureNames[i]]:
                        linedict[featureNames[i]] = float(line[i])
                    else:
                        linedict[featureNames[i]] = line[i]
                        if not line[i] in discreteOptions[featureNames[i]]:
                            print("ERROR discrete class used but not defined in .names file")
                linedict['class'] = line[-1].replace('\n', '')
                data.append(linedict)

    #Make arrays of all the feature values and class values so you can make a dataframe
    feature_values = []
    class_values = []
    for entry in data:
        feature_vals = [v for k, v in entry.items() if k != 'class']
        class_value = entry['class']
        class_values.append(class_value)
        feature_values.append(feature_vals)
    
    df = pd.DataFrame(feature_values)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_features = df[numerical_cols]
    categorical_features = pd.get_dummies(df, columns=categorical_cols) #perform one-hot encoding on categorical data
    scaler = MinMaxScaler() #use the minmaxscaler to scale all the features in the range 0, 1
    scaled_numerical_features = scaler.fit_transform(numerical_features)
    scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=numerical_cols)
    feature_values = pd.concat([scaled_numerical_df, categorical_features.reset_index(drop=True)], axis=1).to_numpy()

    return classes, featureNames, data, iscontinuous, discreteOptions, class_values, feature_values


def fitnessFunction(individual): #The fitness function with 2 fitnesses
    accuracy = wrapper_objective(individual)
    class_percentage = individual.count(1)/len(individual)
    return accuracy, class_percentage

accuracyDict = {}

def wrapper_objective(individual): #use KNN as the wrapper
    if str(individual) in accuracyDict.keys():
        return accuracyDict[str(individual)]
    k = 3

    features = select_features(individual, feature_values, class_values)
    if features is None: #if there are no features
        return 1 - 0.00000001

    X_train, X_test, y_train, y_test = train_test_split(features, class_values, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracyLoss = (1.0 - accuracy)
    accuracyDict[str(individual)]  = accuracyLoss
    return accuracyLoss

def select_features(individual, feature_values, class_values):
    features = None
    datalen = len(class_values)
    for i in range(len(individual)):
        if individual[i] != 0 and individual[i] != 1:
            print("ERROR")
            print(individual)

        if individual[i] == 1:
            if features is None:
                features = [[x] for x in feature_values[:,i]]
            else:
                for index in range(datalen):
                    features[index].append(feature_values[index][i]) 
    # z = pd.DataFrame(features, feat_names)
    return features

toolbox = None

def make_toolbox(mutation_prob):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    NDIM = len(feature_values[0])

    global toolbox
    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NDIM)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitnessFunction)
    toolbox.register("mate", tools.cxOnePoint)

    def mutate_flip_bit(individual, mutation_prob):
        if random.random() < mutation_prob:
            flipindex = random.randint(0, len(individual) - 1)
            individual[flipindex] = abs(individual[flipindex] - 1) #flip from 0 to 1 and vice versa



    toolbox.register("mutate", mutate_flip_bit, mutation_prob=mutation_prob)
    toolbox.register("select", tools.selNSGA2)

def main(seed):
    random.seed(seed)

    num_generations = 20 
    populationCount = 100
    crossover_prob = 0.9
    mutation_prob = 0.1

    make_toolbox(mutation_prob)

    global toolbox

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(populationCount)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # Begin the generational process
    for gen in range(1, num_generations):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= crossover_prob:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(pop), **record)
        print(logbook.stream)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, populationCount)

    return pop, logbook

def display_NSGA2_results(pop, stats):
    pop.sort(key=lambda x: x.fitness.values)

    # Plot the Pareto front
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    obj1 = [fitnessFunction(ind)[0] for ind in front]
    obj2 = [fitnessFunction(ind)[1] for ind in front]
    front_objectives = list(zip(obj1, obj2))
    
    print("Best accuracy: " + str(min(obj1)))
    print("Best class count: " + str(min(obj2)))

    reference_point = np.array([1.1, 1.1])
    hv = hypervolume(front, reference_point)
    print(f'Hypervolume: {hv}')

    print("Pareto front: " + str(front))

    plt.scatter(obj1, obj2)
    plt.title("Pareto Front")
    plt.xlabel("Accuracy Loss (percentage of incorrect identifications)")
    plt.ylabel("Percentage of features selected")
    plt.show()

global classes, featureNames, data, iscontinuous, discreteOptions, class_values, feature_values

def set_global_vars(my_classes, my_featureNames, my_data, my_iscontinuous, my_discreteOptions, my_class_values, my_feature_values):
    global classes, featureNames, data, iscontinuous, discreteOptions, class_values, feature_values
    classes, featureNames, data, iscontinuous, discreteOptions, class_values, feature_values  = my_classes, my_featureNames, my_data, my_iscontinuous, my_discreteOptions, my_class_values, my_feature_values 

if __name__ == "__main__":
    if len(sys.argv) == 2:
        seed = 100
        classes, featureNames, data, iscontinuous, discreteOptions, class_values, feature_values = process_file(sys.argv[1])
        starttime = datetime.datetime.now()
        pop, stats = main(seed)
        print("Time taken: " + str(datetime.datetime.now() - starttime))
        display_NSGA2_results(pop, stats)
    else:
        print("Need another CMD argument, the path to the folder containing data to perform NSGA2 on")
