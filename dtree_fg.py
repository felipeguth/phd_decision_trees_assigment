import math
import sys
import os
import random
from dtree_fg_methods import *

#params
randomFeatureSelection = 1 # 1 = yes 0 = no
featurePercentUsage = 60 #identifies the percentage of attribute usage 60 = 6/10 attributes

ensembleSize = 5
randomBootstrap = 1 #1 = yes 0 = no
percentTraining = 80 #how much used on training set

#Open and read file
op = input("Choose dataset: \n 1 - banks \n 2 - tennis \n 3 - politics \n")

#load data - get current directory and open file
if op == 1:
    path = (os.getcwd() + "/data/banks.csv")
elif op ==2:
    path = (os.getcwd() + "/data/tennis.csv")
elif op == 3:
    path = (os.getcwd() + "/data/politics.csv")

try:
    fileN = open(path, "r")
except IOError:
    print "Error: The file '%s' was not found on this system." % filename
    sys.exit(0)


#Create a dictionary of key values - extract the attribute list in the first row and creates a dictionary of key values

# Create a list of all the lines in the data file
lns = [line.strip() for line in fileN.readlines()]

# create list of attributes
lns.reverse()
attsList = [attr.strip() for attr in lns.pop().split(",")] #extract attribute list
labelClass = attsList[-1]
lns.reverse()

# merge key and values in dictionary
data = []
for line in lns:
    data.append(dict(zip(attsList, [dataLn.strip() for dataLn in line.split(",")])))


#table to combine voting of ensembles
votingTable = []

rowsData = len(data)
trainingSamples = int(round((percentTraining * rowsData) / 100)) #do not treat exceptions of 0 or > 100 percentage


for n in range(0,ensembleSize): #iterates the ensembles
    trainingset = [] #reset array

    # if bootstrap, select random subsample of registers from the original dataset
    if randomBootstrap == 1:

        kArray = []

        for i in range(trainingSamples):
            k = random.randint(0,rowsData-1)
            kArray.append(k) #stores the position of the register k of the bootstrap sample, this position is used later to compare result of classification against true result of labelClass
            trainingset.append(data[k]) #random subsampling of original dataset
    else:
        #does not do random subsampling, just uses percentage of data
        kArray = []
        for i in range(trainingSamples):
            kArray.append(i) #stores the position of the sample k
            trainingset.append(data[i]) #take data

    #controls the feature selection op
    if randomFeatureSelection == 1:
        subsetAtt = []
        numbAtts = int((round((featurePercentUsage * (len(attsList)) - 1) / 100.0)))

        a = (len(attsList)-1)
        listUniqRd = random.sample(xrange(0, a), numbAtts)
         #takes a list of unique random numbers based on a percentage of features to be used (featurePercentUsage) and performs the random feature selection from the original set
        for i in listUniqRd:
            subsetAtt.append(attsList[i])

         #calls method to build decision tree using just the selected features
        instanceTree = buildDT(trainingset, subsetAtt, labelClass, informationGain)

    else:
        #calls method to build decision tree using the entire set of features
        instanceTree = buildDT(trainingset, attsList, labelClass, informationGain)

    #calls method to classify data
    classifierTest = classification(instanceTree, trainingset)

    #takes results of classification and calls method to summarize results
    votingTable.append(zip(kArray,[iResult for iResult in classifierTest]))

itemsTested,ensembleClassif = SummarizeVotes(votingTable)
#print combined results of ensembles
printResults(itemsTested,ensembleClassif,data,labelClass)