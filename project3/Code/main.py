from helpers import helpers as hp
from knn import knn
from naive_bayes import bayes
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        nn = int(input("Enter the number of closest neighbors to consider: "))
        h.normalizeEvaluationSet(predictData, mean, stdDev)
        for i in range(len(trainData)):
            tmp = None
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
        for i in range(len(trainData)):
            tmp = None
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
        maxProb = float('-inf')
        classKey = -1
        for key in probabilities:
            print("P(X|H{})*P(H{}) = {}".format(key,key,probabilities[key]))
            if probabilities[key] > maxProb:
                maxProb = probabilities[key]
                classKey = key
        print("This test data record belongs to: Class {}".format(classKey))

        
        

    def decision_tree(self, kCrossValidation):
        print("\nRunning Decision Tree Classifier ....................\n")
        from decision_tree import decisionTree
        h = hp()
        fileName = h.get_fileName()
        # filePath = "../Data/"+fileName+".txt"
        filePath = "CSE-601/project3/Data/"+fileName+".txt"
        data, labels = h.readData(filePath)
        data = h.oneHotEncoding(data, labels)
        dt = decisionTree()

        accuracy = []
        precision = []
        recall = []
        f_score = []
        models = []

        if kCrossValidation <= 1:
            root = dt.decision(data)
            print(root)
            target = data.iloc[:,-1]
            predicted = dt.predictData(data.iloc[:, :-1], root)
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParameters(predicted, target)
            accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
            tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            precision.append(tmpPrecision)
            recall.append(tmpRecall)
            f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
        else:    
            foldSize = int(data.shape[0] / kCrossValidation)
            for i in range(kCrossValidation):
                print("Running iteration " + str(i+1) + " of k cross validation .....")
                testData = data.loc[foldSize*i:foldSize*(i+1)-1]
                trainData = data.loc[:foldSize*i-1].append(data.loc[foldSize*(i+1):])
                root = dt.decision(trainData)
                target = testData.iloc[:,-1].values.tolist()

                predicted = dt.predictData(testData.iloc[:, :-1], root)
                models.append(root)
                truePositives, trueNegatives, falsePositives, falseNegatives = h.findParameters(predicted, target)
                accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
                tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
                tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
                precision.append(tmpPrecision)
                recall.append(tmpRecall)
                f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))

        print("\nMetrics on train data with k-cross validation")
        h.calculateMetrics(accuracy, precision, recall, f_score)

        fileName = input("\nEnter test data file name without extension (if no test file, just press enter): ")
        if fileName != '':
            filePath = "../Data/"+fileName+".txt"
            # filePath = "CSE-601/project3/Data/"+fileName+".txt"
            testData, testLabels = h.readData(filePath)
            testData = h.oneHotEncoding(testData, testLabels)
            predLabels = []
            for _,row in testData.iloc[:,:-1].iterrows():
                predictedRow = [dt.predictRow(row, root) for root in models]
                predLabels.append(max(set(predictedRow), key=predictedRow.count))
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParameters(predLabels, testData.iloc[:,-1].values.tolist())
            accuracy = [h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives)]
            precision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            recall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            f_score = [h.findFMeasure(precision, recall)]
            print("\nMetrics on test data with bagging")
            h.calculateMetrics(accuracy, [precision], [recall], f_score)

    def random_forest(self, kCrossValidation):
        print("\nRunning Random Forest Classifier ....................\n")
        from random_forest import randomForest
        h = hp()
        fileName = h.get_fileName()
        filePath = "../Data/"+fileName+".txt"
        # filePath = "CSE-601/project3/Data/"+fileName+".txt"
        data, labels = h.readData(filePath)
        data = h.oneHotEncoding(data, labels)
        rf = randomForest()

        try:
            numTrees = int(input("\nEnter number of trees: "))
            numFeatures = int(input("Enter number of features to consider: "))
        except:
            print("\nExecution Failed - Wrong Input")
            exit()

        accuracy = []
        precision = []
        recall = []
        f_score = []
        models = []

        foldSize = int(data.shape[0] / kCrossValidation)
        for i in range(kCrossValidation):
            print("Running iteration " + str(i+1) + " of k cross validation .....")
            testData = data.loc[foldSize*i:foldSize*(i+1)-1]
            trainData = data.loc[:foldSize*i-1].append(data.loc[foldSize*(i+1):])
            forest = rf.forest(trainData, numTrees=numTrees, numFeatures=numFeatures)
            target = testData.iloc[:,-1].values.tolist()
            predicted = rf.predictForest(testData.iloc[:, :-1], forest)
            models.append(forest)
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParameters(predicted, target)
            accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
            tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            precision.append(tmpPrecision)
            recall.append(tmpRecall)
            f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
        
        print("\nMetrics on train data with k-cross validation")
        h.calculateMetrics(accuracy, precision, recall, f_score)

        fileName = input("\nEnter test data file name without extension (if no test file, just press enter): ")
        if fileName != '':
            filePath = "../Data/"+fileName+".txt"
            # filePath = "CSE-601/project3/Data/"+fileName+".txt"
            testData, testLabels = h.readData(filePath)
            testData = h.oneHotEncoding(testData, testLabels)
            predLabels = []
            for forest in models:
                predLabels.append(rf.predictForest(testData, forest))
            predLabels = pd.DataFrame(predLabels)
            pred = []
            for _, colData in predLabels.iteritems():
                pred.append(colData.value_counts().index[0])
            truePositives, trueNegatives, falsePositives, falseNegatives = h.findParameters(pred, testData.iloc[:,-1].values.tolist())
            accuracy = [h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives)]
            precision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
            recall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
            f_score = [h.findFMeasure(precision, recall)]
            print("\nMetrics on test data with bagging")
            h.calculateMetrics(accuracy, [precision], [recall], f_score)

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
    
    elif algorithm == 1:
        print("Enter train File name")
        trainData = h.get_file(h.get_fileName(), kCrossValidation = 10)
        accuracy, precision, recall, f_score = m.knn(predictData, trainData)
        h.calculateMetrics(accuracy, precision, recall, f_score)
    
    elif algorithm == 2:
        m.decision_tree(kCrossValidation=10)
    
    elif algorithm == 3:
        print("Enter train File name")
        trainData = h.get_file_bayes(h.get_fileName(), kCrossValidation = 10)
        accuracy, precision, recall, f_score = m.bayes_naive(predictData, trainData)
        h.calculateMetrics(accuracy, precision, recall, f_score)
    
    elif algorithm == 4:
        print("Enter train File name")
        trainData = h.get_file_bayes_demo(h.get_fileName())
        print("Enter test File name")
        predictData = h.get_file_bayes_demo(h.get_fileName(),fileType = 'predictData')
        m.bayes_naive_demo(predictData, trainData)
    
    elif algorithm == 5:
        m.random_forest(kCrossValidation=10)
    
    else:
        print("\nWrong input")
