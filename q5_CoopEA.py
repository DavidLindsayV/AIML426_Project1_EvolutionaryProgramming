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

def plot_fitness(avg_fitness, min_fitness):
    # Create a range of epochs
    epoch_range = range(1, len(avg_fitness) + 1)

    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, min_fitness, label='Best fitness')
    plt.plot(epoch_range, avg_fitness, label='Average fitness')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Fitness (log)')
    plt.title('Fitness over Epochs')
    plt.yscale('log')
    plt.legend()

    # Show the plot
    plt.show()

def protectedDiv(left, right):
    if right == 0 or right == 0.0:
        return 1.0
    else:
        return left / right

def generate_primitive_set():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    # pset.addPrimitive(if_positive, 3) # No need for if_positive due to knowing that there is only 1 discontinuity and it's been accounted for
    pset.renameArguments(ARG0='x') #this is the input terminal
    pset.addEphemeralConstant("rand101", partial(random.randint, -5, 5))
    return pset


def SymbolicRegression():
    num_train_cases = 100 #100 data inputs generated for evaluation
    num_test_cases = 100 

    pset = generate_primitive_set()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    global toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # def makeSpecies(n):
    #     return [toolbox.population(n), toolbox.population(n)]
    # toolbox.register("species", makeSpecies)

    def calcMSE(individuals):
        indiv1 = individuals[0]
        indiv2 = individuals[1]
        # Transform the tree expression in a callable function
        func1 = toolbox.compile(expr=indiv1)
        func2 = toolbox.compile(expr=indiv2)
        # Evaluate the mean squared error between the expression and the real function
        sqerrors1 = ((func1(x) - target_function(x)) ** 2 for x in np.random.uniform(-100.0, 0, num_train_cases))
        sqerrors2 = ((func2(x) - target_function(x)) ** 2 for x in np.random.uniform(0, 100.0, num_train_cases))
        return (math.fsum(sqerrors1) / num_train_cases + math.fsum(sqerrors2) / num_train_cases,)

    toolbox.register("evaluate", calcMSE)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def get_best(population):
        return sorted(population, key=lambda ind: ind.fitness.values, reverse=True)[:1]

    toolbox.register("get_best", get_best)


    random.seed(123)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    species, representatives = CoopEA()

    test_cases = np.random.uniform(-100.0, 100.0, num_test_cases)
    # func = toolbox.compile(expr=indiv)
    print(representatives)
    print("MSE over " + str(num_test_cases) + " test cases = " +  str(calcMSE(representatives)[0]))

    print("Actual function: ")
    print("if_positive(x, add(add(mul(2, x), mul(x, x)), 3), protectedDiv(1, add(x, sin(x))))")

    # plot_fitness(log.chapters["fitness"].select("avg"), log.chapters["fitness"].select("min"))

toolbox = None

def target_function(val):
    if val <= 0:
        return 2 * val + val * val + 3.0
    else:
        return 1.0 / val + math.sin(val)


def plot_MSE(MSE):
    # Create a range of epochs
    epoch_range = range(1, len(MSE) + 1)

    # Plot the accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, MSE, label='Representative MSE')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.title('Representative MSE over Epochs')
    plt.legend()

    # Show the plot
    plt.show()

def CoopEA():
    global toolbox
    populationSize = 300
    numEpochs = 20
    crossoverProb = 0.6
    mutationProb = 1.0

    species = [toolbox.population(populationSize) for _ in range(2)]
    # print(species)
    representatives = [random.choice(species[i]) for i in range(2)]
    # print(species[0].)

    mses = []

    for epoch in range(numEpochs):
        # Initialize a container for the next generation representatives
        next_repr = [None] * len(species)
        for i, s in enumerate(species):
            # Vary the species individuals
            s = algorithms.varAnd(s, toolbox, crossoverProb, mutationProb)
        
            # Get the representatives excluding the current species
            r = [None, None]
            for j in range(len(r)):
                if j != i:
                    r[j] = representatives[j]

            for ind in s:
                # Evaluate and set the individual fitness
                r[i] = ind 
                ind.fitness.values = toolbox.evaluate([ind] + r)

            # Select the individuals
            species[i] = toolbox.select(s, len(s))  # Tournament selection
            next_repr[i] = toolbox.get_best(s)[0]   # Best selection

        representatives = next_repr
        mses.append(toolbox.evaluate(representatives))
    

    plot_MSE(mses)
    
    return species, representatives

if __name__ == '__main__':
    if len(sys.argv) == 1:
        SymbolicRegression()
    else:
        print('You need no extra cmd arguments')