import numpy as np
import pandas as pd

'''
Decision Tree Regressor
'''

class DecisionTreeClassifier():

    def __init__(self,max_depth,depth=1,min_size=1):
        self.max_depth=max_depth
        self.depth=depth
        self.min_size=min_size

    def fit(self,x,y):
        self.x=x
        self.y=y

        self.train=np.concatenate((x,y),axis=1)
        self.build_tree(self.train,self.max_depth,self.min_size)

    def gini_index(self,groups, classes):
        n_instances = float(sum([len(group) for group in groups]))

        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def test_split(self,index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def to_terminal(self,group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self,node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])

        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return

        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return

        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth+1)

        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth+1)


    def build_tree(self,train, max_depth, min_size):
        self.node = self.get_split(train)
        self.split(self.node, max_depth, min_size, 1)


    def predict(self,x):
        results=np.array([0]*len(x))
        for i,row in enumerate(x):
            results[i]=self._get_prediction(self.node,row)

        return results

    def _get_prediction(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._get_prediction(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._get_prediction(node['right'], row)
            else:
                return node['right']

    def printtree(self, depth=0):
        if isinstance(self.node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (self.node['index']+1), self.node['value'])))
            self.node['left'].printtree( depth+1)
            self.node['right'].printtree( depth+1)
        else:
            print('%s[%s]' % ((depth*' ', self.node)))

'''
Decision Tree Regressor
'''

class DecisionTreeRegressor():

    def __init__(self,max_depth,depth=1,min_size=1):
        self.max_depth=max_depth
        self.depth=depth
        self.min_size=min_size

    def fit(self,x,y):
        self.x=x
        self.y=y

        self.train=np.concatenate((x,y),axis=1)
        self.build_tree(self.train,self.max_depth,self.min_size)

    def mean_squared_error(self,groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        
        error = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            target=[row[-1] for row in group]
            #print(target)
            mean=sum(target)/len(target)
            for class_val in target:
                score += (mean-class_val)**2
        
            #Error is weight sum 
            error+=score/len(group)*(size)/n_instances
        
        return error

    def test_split(self,index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        #print(left,right)
        return left, right

    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 99999999999999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                error = self.mean_squared_error(groups, class_values)
                if error < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], error, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def to_terminal(self,group):
        #Calculaing mean value of group
        outcomes = sum([row[-1] for row in group])/len(group)
        return outcomes

    def split(self,node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
    
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
    
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
    
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth+1)
    
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth+1)
        

    def build_tree(self,train, max_depth, min_size):
        self.node = self.get_split(train)
        self.split(self.node, max_depth, min_size, 1)


    def predict(self,x):
        results=np.array([0]*len(x))

        for i,row in enumerate(x):
            results[i]=self._get_prediction(self.node,row)

        return results

    def _get_prediction(self, node,row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._get_prediction(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._get_prediction(node['right'], row)
            else:
                return node['right']