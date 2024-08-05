import math
import sys
import os
from turtle import pd

import numpy as np
from pandas import DataFrame
import pandas as pd
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
    iscontinuous = {}
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
                iscontinuous[line[0]] = (bool(line[1].replace('\n', '')))

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
            
    return classes, featureNames, data, iscontinuous 


def calculate_conditional_entropy(df, feature_column, target_column):
    total_samples = len(df)
    feature_values_counts = df[feature_column].value_counts().to_dict()
    conditional_entropy = 0.0
    for feature_value, feature_count in feature_values_counts.items():
        # Step 5: Filter the dataframe where the feature column equals the current feature_value
        subset_df = df[df[feature_column] == feature_value]
        
        # Step 6: Get the unique values and their counts for the target column Y in the subset
        target_values_counts = subset_df[target_column].value_counts().to_dict()

        # Step 7: Calculate the probability of the current feature value
        P_X = feature_count / total_samples
        
        # Step 8: Initialize entropy for this subset (conditional on the current feature value)
        subset_entropy = 0.0

        # Step 9: Loop over each unique value of the target column Y
        for target_value, target_count in target_values_counts.items():
            # Step 10: Calculate the conditional probability P(Y|X)
            P_Y_given_X = target_count / feature_count

            # Step 11: Update the subset entropy using the formula -P(Y|X) * log2(P(Y|X))
            subset_entropy -= P_Y_given_X * np.log2(P_Y_given_X)

        # Step 12: Update the overall conditional entropy by weighting the subset entropy by P(X)
        conditional_entropy += P_X * subset_entropy

    # Step 13: Return the final conditional entropy value
    return conditional_entropy


def filter_objective(individual):
    global datadf
    features = select_features(individual, datadf)
    if features.shape(1) == 0:
        return 0.0000000001, True
    
    X_train, X_test, y_train, y_test = train_test_split(features, class_values, test_size=0.3, random_state=42)
    class_entropy = 0
    for cls in classes:
        py = y_train.count(cls) / len(y_train)
        class_entropy += - py * math.log2(py)
    class_entropy = -class_entropy


    
    return None

accuracyDict = {}

def select_features(individual, dataframe):
    datalen = dataframe.shape[0]
    trimmedDF = dataframe.copy()
    for i in range(len(individual)):
        if individual[i] == 0:
            trimmedDF.drop(columns=featurenames[i])
    return trimmedDF

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
    # z = pd.DataFrame(features, feat_names)
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

datadf = None 

def FilterGA():
    global datadf
    datadf = DataFrame(data)
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
    
    #Discretize the data
    for featurename in featurenames:
        if iscontinuous[featurename]:
            datadf[featurename] = pd.qcut(datadf[featurename], q=5)

    best_feasible_score, best_feasible_solution, best_feasible_scores, ave_scores = evolve_population(population, numEpochs, alpha, filter_objective, geneweights, mutation_rate, populationSize, elitism)

    print("Best score = " + str(best_feasible_score))
    print("Best solution = " + str(best_feasible_solution))
    plot_scores(best_feasible_scores, ave_scores)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        classes, featurenames, data, iscontinuous = process_file(sys.argv[1])
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