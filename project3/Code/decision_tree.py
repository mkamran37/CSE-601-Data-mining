import numpy as np
import pandas as pd
from math import log

class decisionTree:

    def readData(self, filePath):
        data = np.genfromtxt(filePath, dtype=None, delimiter="\t", encoding=None)
        dataDf = pd.DataFrame(data)
        labels = dataDf.iloc[:,-1]
        return dataDf.iloc[:,:-1], dataDf.iloc[:,-1]

    def oneHotEncoding(self, data, labels):
        for colName, colData in data.iteritems():
            if colData.dtype == np.object:
                data = pd.concat([data, pd.get_dummies(colData, prefix=colName)], axis=1)
                data.drop([colName], axis=1, inplace=True)

        return pd.concat([data, labels], axis=1)

    def decision(self, data):
        print("Running Decision Tree Classifier ....................")
        root = self.createTree(data.loc[:70*data.shape[0] / 100])
        # print(root)
        testData = data.loc[70*data.shape[0] / 100:]
        target = testData.iloc[:,-1].values.tolist()
        predicted = self.testData(testData.iloc[:, :-1], root)
        return target, predicted

    def createTree(self, data):
        n = Node()

        print(data)
        if data.iloc[:,-1].value_counts().shape[0] == 1:
            n.feature = data.iloc[:, -1].iloc[0]
            return n

        if data.shape[1] == 2:
            n.feature = data.iloc[:,-1].value_counts().index[0]
            return n

        bestFeature = self.getBestFeature(data)
        n.feature = bestFeature

        condition = (data[bestFeature].max() + data[bestFeature].min()) / 2
        n.condition = condition

        leftChildData = data.loc[data[bestFeature] < condition]
        leftChildData = leftChildData.drop(bestFeature, axis=1)
        # print(leftChildData)
        n.left = self.createTree(leftChildData)

        rightChildData = data.loc[data[bestFeature] >= condition]
        rightChildData = rightChildData.drop(bestFeature, axis=1)
        # print(rightChildData)
        n.right = self.createTree(rightChildData)

        return n

    def getBestFeature(self, data):
        entropy_p = self.entropy(data)
        max_gain = float('-inf')
        bestFeature = 0
        for colName, colData in data.iloc[:,:-1].iteritems():
            condition = (colData.max() - colData.min()) / 2
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

        return bestFeature

    def entropy(self, data):
        entropy = 0.0
        labelCounts = data.iloc[:,-1].value_counts()
        for idx in labelCounts.index:
            prob = float(labelCounts[idx]) / len(data)
            entropy -= prob * log(prob, 2)

        return entropy

    def testData(self, data, root):
        predicted = []
        for index, row in data.iterrows():
            predicted.append(self.testRow(row, root))

        return predicted

    def testRow(self, data, root):
        if not root.left and not root.right:
            return root.feature

        if data[root.feature] < root.condition:
            return self.testRow(data, root.left)
        elif data[root.feature] >= root.condition:
            return self.testRow(data, root.right)


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
