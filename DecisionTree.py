import numpy as np
import pandas as pd

'''
Decision Tree Regressor
'''


class DecisionTreeClassifier:

    def __init__ (self, min_samples_split = 2, max_depth = 5, max_features = None, criterion = 'gini'):

        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.tree = {}


    def checkPurity(self, y):
    
        classes = np.unique(y)
        return len(classes) == 1


    def getSplits(self, X):

        splits = {}
        for i in range(X.shape[1]):
            splits[i]=[]
            unique_values = np.unique(X[:, i])
            for j in range(1, len(unique_values)):
                splits[i].append( (unique_values[j] + unique_values[j-1])/2 )
        return splits


    def splitData(self, X, y, column, value):
  
        column_values = X[:, column]
        X_lesser = X[column_values <= value]
        y_lesser = y[column_values <= value]
        X_greater = X[column_values > value]
        y_greater = y[column_values > value]
        return X_lesser, y_lesser, X_greater, y_greater


    def giniIndex(self, y):

        _, counts = np.unique(y, return_counts=True)
        Pi = counts / np.sum(counts)
        return 1 - np.sum( [i**2 for i in Pi] )
    

    def entropy(self, y):
 
        _, counts = np.unique(y, return_counts=True)
        Pi = counts / np.sum(counts)
        return sum(Pi * -np.log2(Pi))
    

    def measureImpurity(self, y_lesser, y_greater, criterion = 'gini'):

        method = {'gini': self.giniIndex, 'entropy': self.entropy}
        impurity = method[criterion]
        n = len(y_lesser)+len(y_greater)
        return len(y_lesser)/n*impurity(y_lesser) + len(y_greater)/n*impurity(y_greater)
    

    def determineBestSplit(self, X, y, splits):
    
        Imp = 1000
        for column in splits:
            for value in splits[column]:
                _, y_lesser, _, y_greater = self.splitData(X, y, column, value)
                currImp = self.measureImpurity(y_lesser, y_greater, self.criterion)
                if(currImp<Imp):
                    Imp = currImp
                    bestSplitColumn = column
                    bestSplitValue = value
        return bestSplitColumn, bestSplitValue
    

    def classify(self, y):
    
        classes, counts = np.unique(y, return_counts=True)
        return classes[counts.argmax()]


    def Classifier(self, X, y,  counter = 0):

        if (self.checkPurity(y)) or (len(y) < self.min_samples_split) or (counter == self.max_depth):
            return self.classify(y)
        
        else:
            counter+=1
            #np.random.seed(0)
            features = np.sort(np.random.choice(X.shape[1], self.max_features, replace = False))
            potentialSplits = self.getSplits(X[:, features])
            splitColumn, splitValue = self.determineBestSplit(X[:, features], y, potentialSplits)
            X_lesser, y_lesser, X_greater, y_greater = self.splitData(X, y, splitColumn, splitValue)

            if len(y_lesser) == 0:
                return self.classify(y_greater)
            elif len(y_greater) == 0:
                return self.classify(y_lesser)
            
            splitCondition = "{} <= {}".format(splitColumn, splitValue)
            subtree = {splitCondition: []}
            
            true = self.Classifier(X_lesser, y_lesser, counter)
            false = self.Classifier(X_greater, y_greater, counter)
            if(true == false):
                return true 
            else:
                subtree[splitCondition].append(true)
                subtree[splitCondition].append(false)
                return subtree


    def determineMaxFeatrues(self, columns):

        if not self.max_features:
            self.max_features = columns
        elif not isinstance(self.max_features, str):
            self.max_features = min(self.max_features, columns)
        else:
            if self.max_features == 'auto' or self.max_features == 'sqrt':
                self.max_features = int(columns**1/2)
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(columns))


    def fit(self, X, y):
        self.determineMaxFeatrues(X.shape[1])
        self.tree = self.Classifier(X, y, 0)
        return self.tree

    def predictOne(self, X, tree = {}):

        if(tree == {}):
            tree = self.tree

        if not isinstance(tree, dict):
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
        return np.array(predictions)


'''
Decision Tree Regressor
'''


class DecisionTreeRegressor:

    def __init__ (self, min_samples_split = 2, max_depth = 5, max_features=None, criterion = 'mse'):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.tree = {}


    def checkPurity(self, y):
    
        classes = np.unique(y)
        return len(classes) == 1
    

    def getSplits(self, X):

        splits = {}
        for i in range(X.shape[1]):
            splits[i]=[]
            unique_values = np.unique(X[:, i])
            for j in range(1, len(unique_values)):
                splits[i].append( (unique_values[j] + unique_values[j-1])/2 )
        return splits
    

    def splitData(self, X, y, column, value):
  
        column_values = X[:, column]
        X_lesser = X[column_values <= value]
        y_lesser = y[column_values <= value]
        X_greater = X[column_values > value]
        y_greater = y[column_values > value]
        return X_lesser, y_lesser, X_greater, y_greater
    
    
    def MeanSquaredError(self, y):
            
        if(len(y) == 0):
            return 0
        yMean = np.mean(y)
        mse = np.mean((y - yMean) **2)
        return mse
    

    def MeanAbsoluteError(self, y):

        if(len(y) == 0):
            return 0
        yMean = np.mean(y)
        mae = np.mean(np.absolute(y - yMean))
        return mae


    def measureError(self, y_lesser, y_greater, criterion='mse'):

        method = {'mse': self.MeanSquaredError, 'mae': self.MeanAbsoluteError}
        error = method[criterion]
        n = len(y_lesser)+len(y_greater)
        return len(y_lesser)/n*error(y_lesser) + len(y_greater)/n*error(y_greater)
    

    def determineBestSplit(self, X, y, splits):
    
        firstIterationFlag = 1
        Imp = 1000
        for column in splits:
            for value in splits[column]:
                _, y_lesser, _, y_greater = self.splitData(X, y, column, value)
                currImp = self.measureError(y_lesser, y_greater, self.criterion)
                if firstIterationFlag == 1 or  currImp < Imp:
                    firstIterationFlag = 0
                    Imp = currImp
                    bestSplitColumn = column
                    bestSplitValue = value
        return bestSplitColumn, bestSplitValue
    

    def findMean(self, y):
        leaf =  np.mean(y)
        if(isinstance(y[0], np.integer)):
            return round(leaf)
        else:
            return leaf


    def Regressor(self, X, y,  counter = 0):

        if (self.checkPurity(y)) or (len(y) < self.min_samples_split) or (counter == self.max_depth):
            return self.findMean(y)
            
        else:
            counter+=1
            #np.random.seed(0)
            features = np.sort(np.random.choice(X.shape[1], self.max_features, replace = False))
            potentialSplits = self.getSplits(X[:, features])
            splitColumn, splitValue = self.determineBestSplit(X[:, features], y, potentialSplits)
            X_lesser, y_lesser, X_greater, y_greater = self.splitData(X, y, splitColumn, splitValue)

            if len(y_lesser) == 0:
                return self.findMean(y_greater)
            elif len(y_greater) == 0:
                return self.findMean(y_lesser)
            
            spitCondition = "{} <= {}".format(splitColumn, splitValue)
            subtree = {spitCondition: []}
            true = self.Regressor(X_lesser, y_lesser, counter)
            false = self.Regressor(X_greater, y_greater, counter)
            if(true == false):
                return true 
            else:
                subtree[spitCondition].append(true)
                subtree[spitCondition].append(false)
                return subtree


    def determineMaxFeatrues(self, columns):

        if not self.max_features:
            self.max_features = columns
        elif not isinstance(self.max_features, str):
            self.max_features = min(self.max_features, columns)
        else:
            if self.max_features == 'auto' or self.max_features == 'sqrt':
                self.max_features = int(columns**1/2)
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(columns))


    def fit(self, X, y):
        self.determineMaxFeatrues(X.shape[1])
        self.tree = self.Regressor(X, y, 0)
        return self.tree

    def predictOne(self, X, tree = {}):

        if(tree == {}):
            tree = self.tree

        if not isinstance(tree, dict):
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
        return np.array(predictions)