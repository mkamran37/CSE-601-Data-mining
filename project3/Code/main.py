from helpers import helpers as hp
from knn import knn
# from naive_bayes import bayes

class main:
    def knn(self, kCrossValidation = 10):
        h = hp()
        k = knn()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        fileName = h.get_fileName()
        trainData = h.get_file(fileName)
        for i in range(len(trainData)):
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
        averageAccuracy = sum(accuracy)/len(accuracy)
        averagePrecision = sum(precision)/len(precision)
        averageRecall = sum(recall)/len(recall)
        averageFscore = sum(f_score)/len(f_score)
        print("ACCURACY = {}%".format(averageAccuracy*100))
        print("PRECISION = {}%".format(averagePrecision*100))
        print("RECALL = {}%".format(averageRecall*100))
        print("F MEASURE = {}%".format(averageFscore*100))

    def decision_tree(self, kCrossValidation = 10):
        from decision_tree import decisionTree
        h = hp()
        fileName = h.get_fileName()
        # filePath = "../Data/"+fileName+".txt"
        filePath = "CSE-601/project3/Data/"+fileName+".txt"
        dt = decisionTree()
        data, labels = dt.readData(filePath)
        data = dt.oneHotEncoding(data, labels)
        target, predicted = dt.decision(data)
        print(target)
        print(predicted)

        
    
    def bayes_naive(self, kCrossValidation = 10):
        h = hp()
        k = knn()
        nb = bayes()
        accuracy = []
        precision = []
        recall = []
        f_score = []
        fileName = h.get_fileName()
        trainData = h.get_file(fileName)
        for i in range(len(trainData)):
            predictData = trainData[i]
            tmp = [lt for j, lt in enumerate(trainData) if j != i]
            td = h.convertToList(tmp)


main().decision_tree()