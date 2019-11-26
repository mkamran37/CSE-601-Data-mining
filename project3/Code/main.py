from helpers import helpers as hp
from knn import knn
from naive_bayes import bayes
from sklearn import preprocessing
import numpy as np

class main:
    def knn(self, predictData = None, trainData = None):
        h = hp()
        k = knn()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        mean, stdDev = h.normalizeData(trainData)
        pd = None
        if predictData is not None:
            h.normalizeEvaluationSet(predictData, mean, stdDev)
            pd = 0
        for i in range(len(trainData)):
            tmp = None
            if pd is None:
                predictData = trainData[i]
            tmp = [lt for j, lt in enumerate(trainData) if j != i]
            td = h.convertToList(tmp)
            k.classify(td, predictData)
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParams(predictData)
            accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
            tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            precision.append(tmpPrecision)
            recall.append(tmpRecall)
            f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
        return accuracy, precision, recall, f_score
    
    def bayes_naive(self, predictData, trainData):
        h = hp()
        nb = bayes()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        pd = None
        if predictData is not None:
            pd = 0
        for i in range(len(trainData)):
            tmp = None
            if pd is None:
                predictData = trainData[i]
            tmp = [lt for j, lt in enumerate(trainData) if j != i]
            td = h.convertToList(tmp)
            classPriorProbabilities = nb.findClassPriorProbability(td)
            classes = nb.segregateClasses(td)
            occurences, means, stdDev = nb.findDescriptorPosteriorProbabilites(classes, td)
            nb.classify(predictData, classPriorProbabilities, occurences, means, stdDev)
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParams(predictData)
            accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
            tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            precision.append(tmpPrecision)
            recall.append(tmpRecall)
            f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
        return accuracy, precision, recall, f_score

if __name__ == "__main__":
    m = main()
    h = hp()
    algorithm = int(input("Enter 1 for K-Nearest Neigbour Algorithm\nEnter 2 for Decision Tree Algorithm\nEnter 3 for Naive Bayes Algorithm\nEnter 4 for Random Forest Algorithm\n"))
    if algorithm == 1:
        print("Enter train File name")
        trainData = h.get_file(h.get_fileName(), kCrossValidation = 10)
        print("Enter test File name(if no test file, just press enter)")
        name = h.get_fileName()
        if name == '':
            predictData = None
        else:
            predictData = h.get_file(name, fileType='predictData')
        accuracy, precision, recall, f_score = m.knn(predictData, trainData)
        h.calculateMetrics(accuracy, precision, recall, f_score)
    elif algorithm == 3:
        print("Enter train File name")
        trainData = h.get_file_bayes(h.get_fileName(), kCrossValidation = 10)
        print("Enter test File name(if no test file, just press enter)")
        name = h.get_fileName()
        if name == '':
            predictData = None
        else:
            predictData = h.get_file_bayes(name, fileType='predictData')
        accuracy, precision, recall, f_score = m.bayes_naive(predictData, trainData)
        h.calculateMetrics(accuracy, precision, recall, f_score)