import numpy as np
import pandas as pd
from math import log
import random

class decisionTree:

    def decision(self, trainData, maxFeatures=None, depth=float('inf'), minLeafRows=0, rf=False):
        features = trainData.columns.values.tolist()
        features.pop()
        root = self.createTree(trainData, features, maxFeatures, depth, minLeafRows, rf)
        return root

    def createTree(self, data, features, maxFeatures, depth, minLeafRows, rf):
        n = Node()

        if data.shape[0] == 0:
            return None

        if depth <= 0 or data.shape[0] <= minLeafRows:
            n.feature = data.iloc[:,-1].value_counts().index[0]
            return n

        if data.iloc[:,-1].value_counts().shape[0] == 1:
            n.feature = data.iloc[:,-1].iloc[0]
            return n

        if len(features) == 0:
            n.feature = data.iloc[:,-1].value_counts().index[0]
            return n

        if rf == True: 
            sampledData = pd.concat([data[random.sample(features, k=maxFeatures)], data.iloc[:,-1]], axis=1)
            bestFeature, condition = self.getBestFeature(sampledData)
        else:
            bestFeature, condition = self.getBestFeature(pd.concat([data[features], data.iloc[:,-1]], axis=1))
            features = [x for _,x in enumerate(features) if x != bestFeature]
        n.feature = bestFeature
        n.condition = condition

        leftChildData = data.loc[data[bestFeature] < condition]
        rightChildData = data.loc[data[bestFeature] >= condition]

        if leftChildData.shape[0] == 0:
            temp = Node()
            temp.feature = data.iloc[:,-1].value_counts().index[0]
            n.left = temp
        else:
            n.left = self.createTree(leftChildData, features, maxFeatures, depth-1, minLeafRows, rf)

        if rightChildData.shape[0] == 0:
            temp = Node()
            temp.feature = data.iloc[:,-1].value_counts().index[0]
            n.right = temp
        else:
            n.right = self.createTree(rightChildData, features, maxFeatures, depth-1, minLeafRows, rf)

        return n

    def getBestFeature(self, data):
        entropy_p = self.entropy(data)
        max_gain = float('-inf')
        bestFeature = 0.0
        bestCondition = 0.0
        for colName, colData in data.iloc[:,:-1].iteritems():
            percent = [0.2, 0.5, 0.8]
            for p in percent:
                condition = (colData.max() - colData.min()) * p
                entropy_i = 0.0
                subData1 = data.loc[data[colName] < condition]
                prob1 = len(subData1) / float(len(data))
                entropy_i += prob1 * self.entropy(subData1)

                subData2 = data.loc[data[colName] >= condition]
                prob2 = len(subData2) / float(len(data))
                entropy_i += prob2 * self.entropy(subData2)

                info_gain = entropy_p - entropy_i
                if info_gain > max_gain:
                    max_gain = info_gain
                    bestFeature = colName
                    bestCondition = condition

        return bestFeature, bestCondition

    def entropy(self, data):
        entropy = 0.0
        labelCounts = data.iloc[:,-1].value_counts()
        for idx in labelCounts.index:
            prob = float(labelCounts[idx]) / len(data)
            entropy -= prob * log(prob, 2)

        return entropy

    def predictData(self, data, root):
        predicted = []
        for index, row in data.iterrows():
            predicted.append(self.predictRow(row, root))

        return predicted

    def predictRow(self, data, root):
        if not root.left and not root.right:
            return root.feature

        if data[root.feature] < root.condition:
            return self.predictRow(data, root.left)
        elif data[root.feature] >= root.condition:
            return self.predictRow(data, root.right)


class Node:

    def __init__(self):
        self.feature = None
        self.left = None
        self.right = None
        self.condition = None

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.feature)+"\n"
        if self.left:
            ret += self.left.__str__(level+1)
        if self.right:
            ret += self.right.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'
