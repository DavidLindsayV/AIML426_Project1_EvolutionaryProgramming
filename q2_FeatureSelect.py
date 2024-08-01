import sys
import os

def process_file(folderpath):
    foldername = folderpath.split(os.path.sep)[-1]
    datafile = os.path.join(folderpath, foldername + ".data")
    namesfile = os.path.join(folderpath, foldername + ".names")

    featureNames = []
    classes = []
    with open(namesfile, 'r') as file:
        firstline = True
        for line in file:
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
            line = line.split(',')
            linedict = {}
            if len(line) != len(featureNames) + 1:
                print("ERROR")
            for i in range(len(line) - 1):
                linedict[featureNames[i]] = float(line[i])
            linedict['class'] = line[-1].replace('\n', '')
            data.append(linedict)
            
    return classes, featureNames, data   

if __name__ == '__main__':
    if len(sys.argv) == 2:
        classes, featurenames, data = process_file(sys.argv[1])
        
    else:
        print('You need to input the path to the folder the GA is to run feature selection on')

#TODO - check this works for discrete features too