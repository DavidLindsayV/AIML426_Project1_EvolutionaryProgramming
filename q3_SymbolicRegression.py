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


def if_positive(val, left, right):
    if val > 0.0:
        return left
    else:
        return right


def generate_primitive_set():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(if_positive, 3)
    pset.renameArguments(ARG0='x') #this is the input terminal
    pset.addEphemeralConstant("rand101", partial(random.randint, -5, 5))
    return pset


def target_function(val):
    if val <= 0:
        return 2 * val + val * val + 3.0
    else:
        return 1.0 / val + math.sin(val)


def generate_random_inputs(n):
    return np.random.uniform(-100.0, 100.0, n)


def SymbolicRegression():
    num_train_cases = 100 #100 data inputs generated for evaluation
    num_test_cases = 100 

    pset = generate_primitive_set()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def calcMSE(individual, points):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression and the real function
        sqerrors = ((func(x) - target_function(x)) ** 2 for x in points)
        return (math.fsum(sqerrors) / len(points),)

    toolbox.register("evaluate", calcMSE, points=generate_random_inputs(num_train_cases))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    random.seed(123)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)

    # print("pop")
    # print(pop)
    # print("log")
    # print(log)
    # print("hof")
    # print(hof)
    test_cases = generate_random_inputs(num_test_cases)
    for indiv in hof.items:
        # func = toolbox.compile(expr=indiv)
        print(indiv)
        print("MSE over " + str(num_test_cases) + " test cases = " +  str(calcMSE(indiv, test_cases)[0]))

    print("Actual function: ")
    print("if_positive(x, add(add(mul(2, x), mul(x, x)), 3), protectedDiv(1, add(x, sin(x))))")

    plot_fitness(log.chapters["fitness"].select("avg"), log.chapters["fitness"].select("min"))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        SymbolicRegression()
        # TODO train GP for sybmolic regression
    else:
        print("Too many CMD arguments")
