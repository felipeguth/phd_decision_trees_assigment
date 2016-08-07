import sys
import math

sys.setrecursionlimit(100000) #solves problems with large number of ensembles

#this method takes as input the attributelist and build a decision tree
def buildDT(data, listAttr, labelClass, gainfunc):
    #print "create decision tree"
    data = data[:]
    valuesClass = [reg[labelClass] for reg in data]
    mostFreqV = mostFreq(data, labelClass)

    if not data or (len(listAttr) - 1) <= 0: #fix prob with banks.csv, return default most frequent
        return mostFreqV

    # if the registers in the data are pure return classification
    elif valuesClass.count(valuesClass[0]) == len(valuesClass):
        return valuesClass[0]
    else:
        # calls function to select the attribute to split the data
        splitAttribute = selectAttribute(data, labelClass, listAttr, gainfunc)

        #initialize a new instance of decision tree using the attribute selected
        newTree = {splitAttribute:{}}

        #create subtree nodes based on the entropy of attributes until the split generate pure items
        for reg in getPureItems(data, splitAttribute):
            subtree = buildDT(matchItems(data, splitAttribute, reg), [att for att in listAttr if att != splitAttribute],labelClass, gainfunc)
            #fill up the tree with subnodes
            newTree[splitAttribute][reg] = subtree

    return newTree


#given the set of attributes this goes through the attributes and select the best one based on the highest information gain in the data
def selectAttribute(data, labelClass, atts, igFunc):
    data = data[:]
    maxGain = 0.0
    selAtt = None

    for reg in atts:
        infGain = igFunc(data, reg, labelClass)
        if (infGain >= maxGain and reg != labelClass):
            selAtt = reg
            maxGain = infGain

    return selAtt


#this method builds a list of values for the label Class and return the most frequent value
def mostFreq(dataset, labelClass):
    data = dataset[:]
    listV = ([i[labelClass] for i in data])

    listV = listV[:]
    topV = 0
    topFreq = None

    for reg in findPureValues(listV):
        if listV.count(reg) > topV:
            topFreq = reg
            topV = listV.count(reg)

    return topFreq



def calcEntropy(data, labelClass):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    countFreqVal = {}
    entropyV = 0.0

    # Calculate the frequency of each of the values in the target attr
    for reg in data:
        if countFreqVal.has_key(reg[labelClass]):
            countFreqVal[reg[labelClass]] += 1.0
        else:
            countFreqVal[reg[labelClass]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for i in countFreqVal.values():
        entropyV += (-i/len(data)) * math.log(i/len(data), 2)

    return entropyV

#this function calculates the IG of spliting the data by a given attribute
def informationGain(data, attribute, labelClass):
    countFreqVal = {}
    regEntropy = 0.0
    attrEntropy = 0.0

    # Calculate frequency of values in label class
    for reg in data:
        if reg[attribute] in countFreqVal:
            countFreqVal[reg[attribute]] += 1.0
        else:
            countFreqVal[reg[attribute]] = 1.0

    #computes the sum of entropy of values by their probability of occurring
    for reg in countFreqVal.keys():
        probAttr = countFreqVal[reg] / sum(countFreqVal.values())
        subs = [i for i in data if i[attribute] == reg]
        regEntropy += probAttr * calcEntropy(subs, labelClass)

    #takes the entropy of attribute and subtracts from the entropy of dataset
    attrEntropy = calcEntropy(data, labelClass)
    attrEntropy = attrEntropy- regEntropy

    return attrEntropy

#given a dataset, this method build a list of values of the chosen attribute and return a list without renduntance
def getPureItems(data, attr):
    data = data[:]
    pureLst = findPureValues([record[attr] for record in data])
    return pureLst


#this method goes through the values of a list and return the unique values
def findPureValues(listV):
    listV = listV[:]
    pureLst = []

    # Cycle through the list and add each value to the unique list only once.
    for reg in listV:
        if pureLst.count(reg) <= 0:
            pureLst.append(reg)

    # Return the list with all redundant values removed.
    return pureLst


#given a set of data this function returns a list of classification by a decision tree previously built
def classification(instTree, data):
    data = data[:]
    resultClassif = []

    for reg in data:
        resultClassif.append(classifyItem(reg, instTree))

    return resultClassif



def classifyItem(item, dtree):

    if isinstance(dtree, basestring): #if string, leaf node; return answer
        return dtree

    # if not string, continues going trough nodes until reach a string value
    else:
        att = dtree.keys()[0]
        tree = dtree[att][item[att]]
        return classifyItem(item, tree)


#this method performs the matching of registers that have a match for a given value in an attribute and returns a list of them
def matchItems(data, item, value):
    data = data[:]
    matchList = []

    if not data:
        return matchList
    else:
        reg = data.pop()
        if reg[item] == value:
            matchList.append(reg)
            matchList.extend(matchItems(data, item, value))
            return matchList
        else:
            matchList.extend(matchItems(data, item, value))
            return matchList


#this method makes the combination of votes of ensembles
def SummarizeVotes(votingTable):
    vetClas = []
    vetReg = []

    #split the data from the tuples into 2 ordered arrays
    for j in range(len(votingTable)):
        listRegTested = [x[0] for x in votingTable[j]]
        listClass = [x[1] for x in votingTable[j]]

        for k in range(len(listRegTested)):
            vetReg.append(listRegTested[k])
            vetClas.append(listClass[k])

    #find the items that were used in the bootstrap
    listUniqReg = findPureValues(vetReg)

    labelClassOp = []
    labelClassOpCont = []
    vetClasFn = []
    # given a list of items tested (listUniqReg) takes all data entries of the ensembles and check the classification results for these items and sum results of same class values
    for current in listUniqReg:
        for i in range(len(vetReg)):
            c = vetReg[i]
            if c == current:
                classResItem = vetClas[i]
                if len(labelClassOp) >= 1:
                    for j in range(len(labelClassOp)): #goes through the classification values, case there is none: the current is appended, case already exists: the class value receives +1
                        opCurr = labelClassOp[j]
                        if classResItem == opCurr:
                            labelClassOpCont[j] += 1
                        else:
                            labelClassOp.append(classResItem)
                            a = 1
                            labelClassOpCont.append(a)
                else:
                    labelClassOp.append(classResItem)
                    a = 1
                    labelClassOpCont.append(a)

        a=0
        p = 0
        for w in labelClassOpCont: #check which class label value got more votes based on the ensemble results
            if w > a:
                winnerClass = labelClassOp[p]
                a = w
            p +=1

        vetClasFn.append(winnerClass)
        labelClassOp = []
        labelClassOpCont = []
        p=0

    #return the list of tested subset records and the final voting classification for each one. i.e  (reg=1, classwin=yes)
    return listUniqReg, vetClasFn


def printResults(testIt, ensembleClassif, data, labelClass):
    #compare with data
    accur = 0
    for i in range(len(testIt)):
        iData = testIt[i]
        trueClass = data[iData][labelClass]
        print ("ensemble classification item %d = %s. True Classification = %s " % (iData,ensembleClassif[i],trueClass))

        if trueClass == ensembleClassif[i]:
            accur +=1
    perAcc = float((float(accur) / float(len(testIt)))*100.0)
    print ("total accuracy %.2f " % perAcc)