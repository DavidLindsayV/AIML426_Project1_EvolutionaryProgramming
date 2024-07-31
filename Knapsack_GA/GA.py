from random import randint
import sys
import numpy as np
import time

def process_file(filename):
    numitems = -1
    capacity = -1
    items = []
    with open(filename, 'r') as file:
        firstline = True
        for line in file:
            words = line.split()
            if firstline:
                numitems = words[0]
                capacity = words[1]
                firstline = False       
            else:     
                items.append( {'value': words[0], 'weight':words[1]})



    

if __name__ == '__main__':
    if len(sys.argv) == 2:
        capacity, items = process_file(sys.argv[1])
        # GA(capacity, items)
    else:
        print('You need to input the path to the knapsack file to run')