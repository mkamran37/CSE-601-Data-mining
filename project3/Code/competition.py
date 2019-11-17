from helpers import helpers as hp
from knn import knn
from naive_bayes import bayes
from sklearn import preprocessing
from competition_helpers import competition as cp
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix

class main():
    def knn(self, predictData = None, trainData = None, kCrossValidation = 10):
        h = hp()
        matrix = defaultdict(list)
        mean, stdDev = h.normalizeData(trainData)
        h.normalizeEvaluationSet(predictData, mean, stdDev)
        for i in range(len(trainData)):
            tmp = [lt for j, lt in enumerate(trainData) if j != i]
            td = h.convertToList(tmp)
            knn().classify(td, predictData)
            for point in predictData:
                matrix[point.id].append(point.label)
        labels = defaultdict(list)
        for key in matrix:
            labels[key] = 0 if matrix[key].count(0) > matrix[key].count(1) else 1
        for point in predictData:
            point.label = labels[point.id]

    def logisticRegression(self, predictData, trainData, labels):
        h = hp()
        mean, stdDev = h.normalizeData(trainData)
        h.normalizeEvaluationSet(predictData, mean, stdDev)
        finalAnswer = defaultdict(list)
        tmp = list()
        tmpLabels = list()
        for j, lt in enumerate(trainData):
            for k,point in enumerate(lt):
                tmp.append(point.point)
                tmpLabels.append(labels[k])
        pd = list()
        for point in predictData:
            pd.append(point.point)
        lr = LogisticRegression()
        lr.fit(tmp, np.array(tmpLabels))
        y_pred = lr.predict(np.array(pd))
        k = 0
        for i in range(418, 796):
            finalAnswer[i].append(y_pred[k])
            k+=1
        return finalAnswer
    def svm(self, predictData, trainData, labels):
        h = hp()
        matrix = defaultdict(list)
        mean, stdDev = h.normalizeData(trainData)
        h.normalizeEvaluationSet(predictData, mean, stdDev)
        finalAnswer = defaultdict(list)
        tmp = list()
        tmpLabels = list()
        for j, lt in enumerate(trainData):
            for k,point in enumerate(lt):
                tmp.append(point.point)
                tmpLabels.append(float(labels[k]))
        pd = list()
        for point in predictData:
            pd.append(point.point)
        clf = svm.SVR()
        clf.fit(np.array(tmp), np.array(tmpLabels))
        y_pred = clf.predict(np.array(pd))
        k = 0
        for i in range(418, 796):
            finalAnswer[i].append(y_pred[k])
            k+=1
        return finalAnswer
    
    def knn2(self, predictData, trainData, labels):
        h = hp()
        # matrix = defaultdict(list)
        mean, stdDev = h.normalizeData(trainData)
        h.normalizeEvaluationSet(predictData, mean, stdDev)
        finalAnswer = defaultdict(list)
        tmp = list()
        tmpLabels = list()
        for j, lt in enumerate(trainData):
            for k,point in enumerate(lt):
                tmp.append(point.point)
                tmpLabels.append(float(labels[k]))
        pd = list()
        for point in predictData:
            pd.append(point.point)
        knn = KNeighborsClassifier(n_neighbors=6, metric='euclidean')
        knn.fit(np.array(tmp), np.array(tmpLabels))
        y_pred = knn.predict(np.array(pd))
        k = 0
        for i in range(418, 796):
            finalAnswer[i].append(y_pred[k])
            k+=1
        return finalAnswer


if __name__ == "__main__":
    m = main()
    h = hp()
    c = cp()
    print("Enter train File name")
    trainData = c.get_file_competition(h.get_fileName())
    print("Enter the labels file")
    # c.assign_labels(trainData, c.read_labels("../Data/"+h.get_fileName()+".csv"))
    labels = c.read_labels("../Data/"+h.get_fileName()+".csv")
    print("Enter the data to be predicted")
    name = h.get_fileName()
    predictData = c.get_file_competition(name, fileType='predictData')
    # m.knn(predictData, trainData)
    fa = m.logisticRegression(predictData, trainData, labels)
    # fa = m.svm(predictData, trainData, labels)
    # fa = m.knn2(predictData, trainData, labels)
    c.writeToCSV(fa)
    