import numpy as np
import pandas as pd
from decision_tree import decisionTree as dt

class randomForest:

    def forest(self, trainData, numTrees=5, numFeatures=None, numRows=None, maxDepth=10, minLeafRows=5, randomSeed=12):
        if numFeatures == None:
            numFeatures = int(np.sqrt(trainData.shape[1]))

        if numRows == None:
            # numRows = int(trainData.shape[0] * 0.8)
            numRows = trainData.shape[0]

        forest = [self.createForest(trainData.iloc[:, :-1], trainData.iloc[:,-1], numFeatures, numRows, maxDepth, minLeafRows, randomSeed) for i in range(numTrees)]
        return forest

        
    def createForest(self, trainData, labels, numFeatures, numRows, maxDepth, minLeafRows, randomSeed):

        trainData = trainData.sample(numFeatures, axis=1, random_state=randomSeed, replace=False)
        trainData = pd.concat([trainData, labels], axis=1)
        trainData = trainData.sample(numRows, axis=0, random_state=randomSeed, replace=False)

        return dt().createTree(trainData, maxDepth, minLeafRows)

    def predictForest(self, testData, forest):
        predicted = []
        for _, row in testData.iterrows():
            predictedRow = [dt().predictRow(row, root) for root in forest]
            predicted.append(max(set(predictedRow), key=predictedRow.count))
        return predicted




