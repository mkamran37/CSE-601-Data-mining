from helpers import helpers as hp
from knn import knn
from naive_bayes import bayes
from sklearn import preprocessing
import numpy as np

class main:
    def knnDemo(self, predictData = None, trainData = None):
        h = hp()
        k = knn()
        nn = int(input("Enter the number of closest neighbors to consider: "))
        k.classify(trainData, predictData, nn)
        truePositives, trueNegatives, falsePositives, falseNegatives = h.findParams(predictData)
        accuracy = h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives)
        precision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
        recall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
        f_score = h.findFMeasure(precision, recall)
        return accuracy, precision, recall, f_score

    def knn(self, predictData = None, trainData = None):
        h = hp()
        k = knn()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        mean, stdDev = h.normalizeData(trainData)
        pd = None
        nn = int(input("Enter the number of closest neighbors to consider: "))
        if predictData is not None:
            h.normalizeEvaluationSet(predictData, mean, stdDev)
            pd = 0
        for i in range(len(trainData)):
            tmp = None
            if pd is None:
                predictData = trainData[i]
            tmp = [lt for j, lt in enumerate(trainData) if j != i]
            td = h.convertToList(tmp)
            k.classify(td, predictData,nn)
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

    def bayes_naive_demo(self, predictData, trainData):
        h = hp()
        nb = bayes()
        classPriorProbabilities = nb.findClassPriorProbability(trainData)
        classes = nb.segregateClasses(trainData)
        occurences, means, stdDev = nb.findDescriptorPosteriorProbabilites(classes, trainData)
        probabilities = nb.classify_demo(predictData, classPriorProbabilities, occurences, means, stdDev)
        for key in probabilities:
            print("P(X|H{})*P(H{}) = {}".format(key,key,probabilities[key]))

if __name__ == "__main__":
    m = main()
    h = hp()
    algorithm = int(input("Enter 0 to run K-Nearest Neighbors in demo mode\nEnter 1 for K-Nearest Neigbour Algorithm\nEnter 2 for Decision Tree Algorithm\nEnter 3 for Naive Bayes Algorithm\nEnter 4 to run Naive Bayes Algorithm in demo mode\nEnter 5 for Random Forest Algorithm\n"))
    
    if algorithm == 0:
        print("Enter train File name")
        trainData = h.get_file_demo(h.get_fileName())
        print("Enter test File name")
        predictData = h.get_file_demo(h.get_fileName(), fileType='predictData')
        accuracy, precision, recall, f_score = m.knnDemo(predictData, trainData)
        h.calculateMetricsDemo(accuracy, precision, recall, f_score)

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
    
    elif algorithm == 4:
        print("Enter train File name")
        trainData = h.get_file_bayes_demo(h.get_fileName())
        print("Enter test File name")
        predictData = h.get_file_bayes_demo(h.get_fileName(),fileType = 'predictData')
        m.bayes_naive_demo(predictData, trainData)
