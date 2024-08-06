import math
from random import randint, random, choices
import sys
from matplotlib import pyplot as plt
import numpy as np
import time
import heapq
from deap import gp, operator

from sklearn.model_selection import train_test_split

from GAfuncs import evolve_population, plot_scores #evolve_population, plot_scores   

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
def if_positive(val, left, right):
    if val > 0:
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
    pset.addPrimitive(if_positive, 3)
    #TODO add constants


def generate_train_test_data(n_test_cases):
    X = np.random.uniform(-100, 100, n_test_cases)
    y = []
    for val in X:
        if val <= 0:
            y.append(2*val + val*val + 3.0)
        else:
            y.append(1/val + math.sin(val))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #TODO calculate MSE from GP to train/test data

def objective():
    myOutput = 0


def SymbolicRegression():
    x = 1 #TODO

if __name__ == '__main__':
    if len(sys.argv) == 1:
        generate_train_test_data()
        # SymbolicRegression(capacity, items)
        #TODO train GP for sybmolic regression
    else:
        print('Too many CMD arguments')