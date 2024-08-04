import sys
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy

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

def filter_objective():
    #TODO
    x = 1

accuracyDict = {}

def select_features(individual, feature_values, class_values):
    features = None
    datalen = len(class_values)
    for i in range(len(individual)):
        if individual[i] == 1:
            if features is None:
                features = [[x] for x in feature_values[:,i]]
                # print(features)
            else:
                for index in range(datalen):
                    features[index].append(feature_values[index][i]) 
    return features

def wrapper_objective(individual): #use KNN as the wrapper
    if str(individual) in accuracyDict.keys():
        return accuracyDict[str(individual)], True
    k = 3

    features = select_features(individual, feature_values, class_values)
    if features is None: #if there are no features
        return 0.0000000001, True

    X_train, X_test, y_train, y_test = train_test_split(features, class_values, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracyDict[str(individual)]  = accuracy
    return accuracy, True

def WrapperGA():
    #hyperparameters
    populationSize = 100
    numEpochs = 40
    alpha = 5 # a -1 means invalid solution = 0
    mutation_rate = 0.2
    elitism =  0.05
    geneweights = []
    for i in range(len(featurenames)):
        geneweights.append(1)

    # population = [[randint(0, 1) for x in range(numitems)] for _ in range(populationSize)]
    population = [[0 for x in range(len(featurenames))] for _ in range(populationSize)]

    best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = evolve_population(population, numEpochs, alpha, wrapper_objective, geneweights, mutation_rate, populationSize, elitism)

    print("Best score = " + str(best_feasible_score))
    print("Best solution = " + str(best_feasible_solution))
    plot_scores(best_feasible_scores, ave_scores)

def FilterGA():
    #hyperparameters
    populationSize = 100
    numEpochs = 40
    alpha = 5 # a -1 means invalid solution = 0
    mutation_rate = 0.2
    elitism =  0.05
    geneweights = []
    for i in range(len(featurenames)):
        geneweights.append(1)

    # population = [[randint(0, 1) for x in range(numitems)] for _ in range(populationSize)]
    population = [[0 for x in range(len(featurenames))] for _ in range(populationSize)]

    class_entropy = entropy(np.bincount(class_values), base=2)

    best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = evolve_population(population, numEpochs, alpha, filter_objective, geneweights, mutation_rate, populationSize, elitism)

    print("Best score = " + str(best_feasible_score))
    print("Best solution = " + str(best_feasible_solution))
    plot_scores(best_feasible_scores, ave_scores)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        classes, featurenames, data = process_file(sys.argv[1])
        feature_values = []
        class_values = []
        for entry in data:
            feature_vals = [v for k, v in entry.items() if k != 'class']
            class_value = entry['class']
            class_values.append(class_value)
            feature_values.append(feature_vals)
        
        # print(class_values)
        # print(feature_values)
        #normalise features
        # feature_values = np.array(feature_values)
        scaler = MinMaxScaler()
        # Fit and transform the features
        feature_values = scaler.fit_transform(feature_values)

        # print(feature_values)

        FilterGA()
        # WrapperGA()
    else:
        print('You need to input the path to the folder the GA is to run feature selection on')

#TODO
# - check this works for discrete features too
# - ensure your code clearly has all the 3 requirements (listed below) met for both Filter and Wrapper
    # - 1 Feature selection that selects a feature subset from a given dataset. 
    # - 2. Data transformation that takes the original dataset and the selected features, and creates a new dataset with only the selected features (the corresponding subset of columns).
    # - 3. Classification that trains a model to classify the instances of a given dataset (the original or that with only the selected features).
# - have both FilterGA and WrapperGA chosen by command line arguments
# - complete filter-based fitness func