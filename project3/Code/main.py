from helpers import helpers as hp
from knn import knn
from naive_bayes import bayes
from sklearn import preprocessing
import numpy as np

class main:
    def knn(self, trainData, kCrossValidation = 10):
        h = hp()
        k = knn()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        for i in range(len(trainData)):
            predictData = trainData[i]
            tmp = [lt for j, lt in enumerate(trainData) if j != i]
            # h.normalizeData(tmp)
            # h.normalizeEvaluationSet(predictData)
            td = h.convertToList(tmp)
            classes = bayes().segregateClasses(td)
            tp = 1
            tn = 0
            if len(classes[tp]) < len(classes[tn]):
                tn = 1
                tp = 0
            k.classify(td, predictData)
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParams(predictData, tp, tn)
            accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
            tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            precision.append(tmpPrecision)
            recall.append(tmpRecall)
            f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
        return accuracy, precision, recall, f_score
    
    def bayes_naive(self, trainData, kCrossValidation = 10):
        h = hp()
        nb = bayes()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        for i in range(len(trainData)):
            predictData = trainData[i]
            tmp = [lt for j, lt in enumerate(trainData) if j != i]
            h.normalizeData(tmp)
            h.normalizeEvaluationSet(predictData)
            td = h.convertToList(tmp)
            classPriorProbabilities = nb.findClassPriorProbability(td)
            classes = nb.segregateClasses(td)
            tp = 1
            tn = 0
            if len(classes[tp]) < len(classes[tn]):
                tn = 1
                tp = 0
            descriptorPosteriorProbabilites = nb.findDescriptorPosteriorProbabilites(classes)
            nb.classify(predictData, classPriorProbabilities, descriptorPosteriorProbabilites)
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParams(predictData, tp, tn)
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
    trainData = h.get_file(h.get_fileName())
    # accuracy, precision, recall, f_score = m.knn(trainData)
    accuracy, precision, recall, f_score = m.bayes_naive(trainData)
    h.calculateMetrics(accuracy, precision, recall, f_score)

