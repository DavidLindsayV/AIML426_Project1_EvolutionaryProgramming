import sys
import os

from GAfuncs import evolve_population, plot_scores

def process_file(folderpath):
    foldername = folderpath.split(os.path.sep)[-1]
    datafile = os.path.join(folderpath, foldername + ".data")
    namesfile = os.path.join(folderpath, foldername + ".names")

    featureNames = []
    classes = []
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

    data = []
    with open(datafile, 'r') as file:
        for line in file:
            line = line.replace('\r', '')
            line = line.split(',')
            linedict = {}
            if len(line) != len(featureNames) + 1:
                print("ERROR")
            for i in range(len(line) - 1):
                linedict[featureNames[i]] = float(line[i])
            linedict['class'] = line[-1].replace('\n', '')
            data.append(linedict)
            
    return classes, featureNames, data   

def objective():
    #TODO
    x = 1

def FeatureSelect(classes, featurenames, data):
    #hyperparameters
    populationSize = 300
    numEpochs = 500
    alpha = 5 # a -1 means invalid solution = 0
    mutation_rate = 0.2
    elitism =  0.05
    geneweights = []
    for i in range(len(featurenames)):
        geneweights.append(1)

    # population = [[randint(0, 1) for x in range(numitems)] for _ in range(populationSize)]
    population = [[0 for x in range(len(featurenames))] for _ in range(populationSize)]

    best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = evolve_population(population, numEpochs, alpha, objective, geneweights, mutation_rate, populationSize, elitism)

    print("Best score = " + str(best_feasible_score))
    print("Best solution = " + str(best_feasible_solution))
    plot_scores(best_feasible_scores, ave_scores)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        classes, featurenames, data = process_file(sys.argv[1])
        FeatureSelect(classes, featurenames, data)
    else:
        print('You need to input the path to the folder the GA is to run feature selection on')

#TODO
# - check this works for discrete features too
# - 1 Feature selection that selects a feature subset from a given dataset.
# - 2. Data transformation that takes the original dataset and the selected features, and creates a new dataset with only the selected features (the corresponding subset of columns).
# - 3. Classification that trains a model to classify the instances of a given dataset (the original or that with only the selected features).
# - you need to write an objective function (two, actually - filter and wrapper)