from helpers import helpers as hp
from knn import knn
from naive_bayes import bayes
from sklearn import preprocessing
import numpy as np
import pandas as pd

class main:
    def knn(self, predictData = None, trainData = None, kCrossValidation = 10):
        h = hp()
        k = knn()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        pd = predictData
        for i in range(len(trainData)):
            tmp = None
            if predictData == None:
                predictData = trainData[i]
                tmp = [lt for j, lt in enumerate(trainData) if j != i]
            else:
                tmp = trainData
            # h.normalizeData(tmp)
            # h.normalizeEvaluationSet(predictData)
            td = h.convertToList(tmp)
            classes = bayes().segregateClasses(td)
            k.classify(td, predictData)
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParams(predictData)
            if truePositives < trueNegatives:
                truePositives, trueNegatives, falsePositives, falseNegatives = trueNegatives, truePositives, falseNegatives, falsePositives
            accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
            tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            precision.append(tmpPrecision)
            recall.append(tmpRecall)
            f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
            predictData = pd 
        return accuracy, precision, recall, f_score

    def decision_tree(self, kCrossValidation = 10):
        print("\nRunning Decision Tree Classifier ....................\n")
        from decision_tree import decisionTree
        h = hp()
        fileName = h.get_fileName()
        # filePath = "../Data/"+fileName+".txt"
        filePath = "CSE-601/project3/Data/"+fileName+".txt"
        dt = decisionTree()
        data, labels = dt.readData(filePath)
        data = dt.oneHotEncoding(data, labels)

        accuracy = []
        precision = []
        recall = []
        f_score = []
        models = []

        foldSize = int(data.shape[0] / kCrossValidation)
        for i in range(kCrossValidation):
            print("Running iteration " + str(i+1) + " of k cross validation")
            testData = data.loc[foldSize*i:foldSize*(i+1)-1]
            trainData = data.loc[:foldSize*i-1].append(data.loc[foldSize*(i+1):])
            target, predicted, root = dt.decision(trainData, testData)
            models.append(root)
            truePositives, trueNegatives, falsePositives, falseNegatives = dt.findParams(predicted, target)
            # if truePositives < trueNegatives:
            #     truePositives, trueNegatives, falsePositives, falseNegatives = trueNegatives, truePositives, falseNegatives, falsePositives
            accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
            tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            precision.append(tmpPrecision)
            recall.append(tmpRecall)
            f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
        return accuracy, precision, recall, f_score
        
    
    def bayes_naive(self, predictData, trainData, kCrossValidation = 10):
        h = hp()
        nb = bayes()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        pd = predictData
        for i in range(len(trainData)):
            tmp = None
            if predictData == None:
                predictData = trainData[i]
                tmp = [lt for j, lt in enumerate(trainData) if j != i]
            else:
                tmp = trainData
            h.normalizeData(tmp)
            h.normalizeEvaluationSet(predictData)
            td = h.convertToList(tmp)
            classPriorProbabilities = nb.findClassPriorProbability(td)
            classes = nb.segregateClasses(td)
            descriptorPosteriorProbabilites = nb.findDescriptorPosteriorProbabilites(classes)
            nb.classify(predictData, classPriorProbabilities, descriptorPosteriorProbabilites)
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParams(predictData)
            # print(truePositives, trueNegatives, falsePositives, falseNegatives)
            if truePositives < trueNegatives:
                truePositives, trueNegatives, falsePositives, falseNegatives = trueNegatives, truePositives, falseNegatives, falsePositives
            accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
            tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            precision.append(tmpPrecision)
            recall.append(tmpRecall)
            f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
            predictData = pd
        return accuracy, precision, recall, f_score

if __name__ == "__main__":
    m = main()
    h = hp()
    # trainData = h.get_file(h.get_fileName())
    # name = h.get_fileName()
    # if name == '':
    #     predictData = None
    # else:
    #     predictData = h.get_file(name, fileType='predictData')
    # accuracy, precision, recall, f_score = m.knn(predictData, trainData)
    # accuracy, precision, recall, f_score = m.bayes_naive(predictData, trainData)
    # h.calculateMetrics(accuracy, precision, recall, f_score)

    accuracy, precision, recall, f_score = m.decision_tree()
    print(accuracy, precision, recall, f_score)
    h.calculateMetrics(accuracy, precision, recall, f_score)

