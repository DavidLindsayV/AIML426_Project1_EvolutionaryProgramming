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
from scipy.stats import mstats
import os
from sklearn.model_selection import train_test_split

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

    return classes, featureNames, data, iscontinuous, discreteOptions

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

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

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_LOW, BOUND_UP = 0.0, 1.0

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 30

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def main(seed=None):
    random.seed(seed)

    NGEN = 250
    MU = 100
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook

if __name__ == "__main__":
    if len(sys.argv) == 2:
        classes, featureNames, data, iscontinuous, discreteOptions = process_file(sys.argv[1])
        # with open("pareto_front/zdt1_front.json") as optimal_front_data:
        #     optimal_front = json.load(optimal_front_data)
        # Use 500 of the 1000 points in the json file
        # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

        # pop, stats = main()
        # pop.sort(key=lambda x: x.fitness.values)

        # print(stats)
        # print("Convergence: ", convergence(pop, optimal_front))
        # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

        # import matplotlib.pyplot as plt
        # import numpy

        # front = numpy.array([ind.fitness.values for ind in pop])
        # optimal_front = numpy.array(optimal_front)
        # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
        # plt.scatter(front[:,0], front[:,1], c="b")
        # plt.axis("tight")
        # plt.show()
    else:
        print("Need another CMD argument, the path to the folder containing data to perform NSGA2 on")


#TODO make this work with both Clean1 input data type and musk input data type