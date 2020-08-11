import numpy as np
import pandas as pd

"""
# Helper Functions
"""

def giniIndex(y):

    _, counts = np.unique(y, return_counts=True)
    Pi = counts / np.sum(counts)
    return 1 - np.sum( [i**2 for i in Pi] )


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    Pi = counts / np.sum(counts)
    return sum(Pi * -np.log2(Pi))


def measureImpurity(y_lesser, y_greater, criterion = 'gini' ):

    method = {'gini': giniIndex, 'entropy': entropy}
    impurity = method[criterion]
    n = len(y_lesser)+len(y_greater)
    return len(y_lesser)/n*impurity(y_lesser) + len(y_greater)/n*impurity(y_greater)

def checkPurity(y):
    
    classes = np.unique(y)
    return len(classes) == 1


def classify(y):
    
    classes, counts = np.unique(y, return_counts=True)
    return classes[counts.argmax()]

def getSplits(X):
    
    splits = {}
    for i in range(X.shape[1]):
        splits[i]=[]
        unique_values = np.unique(X[:, i])
        for j in range(1, len(unique_values)):
            splits[i].append( (unique_values[j] + unique_values[j-1])/2 )
    return splits
    

def splitData(X, y, column, value):
    
    column_values = X[:, column]
    X_lesser = X[column_values <= value]
    y_lesser = y[column_values <= value]
    X_greater = X[column_values > value]
    y_greater = y[column_values > value]
    return X_lesser, y_lesser, X_greater, y_greater


def determineBestSplit(X, y, splits):
    
    Imp = 1000
    for column in splits:
        for value in splits[column]:
            _, y_lesser, _, y_greater = splitData(X, y, column, value)
            currImp = measureImpurity(y_lesser, y_greater)
            if(currImp<Imp):
                Imp = currImp
                bestSplitColumn = column
                bestSplitValue = value
    return bestSplitColumn, bestSplitValue       



"""
# Decison Tree Class
"""

class DecisionTreeClassifier:

    def __init__ (self, min_samples_split, max_depth):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = {}

    def Classifier(self, X, y,  counter = 0):

        if (checkPurity(y)) or (len(y) < self.min_samples_split) or (counter == self.max_depth):
            return classify(y)
        
        else:
            counter+=1

            potentialSplits = getSplits(X)
            splitColumn, splitValue = determineBestSplit(X, y, potentialSplits)
            X_lesser, y_lesser, X_greater, y_greater = splitData(X, y, splitColumn, splitValue)

            spitCondition = "{} <= {}".format(splitColumn, splitValue)
            subtree = {spitCondition: []}
            true = self.Classifier(X_lesser, y_lesser, counter)
            false = self.Classifier(X_greater, y_greater, counter)
            if(true == false):
                return true 
            else:
                subtree[spitCondition].append(true)
                subtree[spitCondition].append(false)
                return subtree
    
    def fit(self, X, y):
        self.tree = self.Classifier(X, y, 0)


    def predictOne(self, X, tree = {}):

        if(tree == {}):
            tree = self.tree

        if isinstance(tree, str):
            return tree
        
        splitCondition = list(tree.keys())[0]
        splitQn = splitCondition.split(' ')
        splitColumn = int(splitQn[0])
        splitValue = float(splitQn[2])

        if X[splitColumn] <= splitValue:
            answer = tree[splitCondition][0] 
        else:        
            answer = tree[splitCondition][1]

        return self.predictOne(X, answer)

    def predict(self, X):

        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.predictOne(X[i]))
        return predictions

