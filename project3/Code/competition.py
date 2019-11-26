from helpers import helpers as hp
from knn import knn
from naive_bayes import bayes
from sklearn import preprocessing
from competition_helpers import competition as cp
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import normalize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB

class main():
    def logisticRegression(self, predictData, trainData, labels):
        h = hp()
        # mean, stdDev = h.normalizeData(trainData)
        # h.normalizeEvaluationSet(predictData, mean, stdDev)
        finalAnswer = defaultdict(list)
        pd = list()
        matrix = defaultdict(list)
        for point in predictData:
            pd.append(point.point)
        for i in range(len(trainData)):
            tmp = list()
            tmpLabels = list()
            for j, lt in enumerate(trainData):
                if j != i:
                    for point in lt:
                        tmp.append(point.point)
                        tmpLabels.append(labels[point.id])
            pca = PCA(svd_solver='full')
            pca_matrix = pca.fit_transform(tmp)
            pca1_matrix = pca.transform(pd)
            lr = LogisticRegression(solver='sag', max_iter=1500)
            lr.fit(pca_matrix, np.array(tmpLabels))
            y_pred = lr.predict(np.array(pca1_matrix))
            k = 0
            for d in range(418, 796):
                finalAnswer[d].append(y_pred[k])
                k+=1
        for key in finalAnswer:
            matrix[key] = 0 if finalAnswer[key].count(0) > finalAnswer[key].count(1) else 1
        return matrix
   
    def svm(self, predictData, trainData, labels):
        h = hp()
        matrix = defaultdict(list)
        finalAnswer = defaultdict(list)
        pca = PCA(n_components=25)
        for i in range(len(trainData)):
            tmp = list()
            tmpLabels = list()
            pd = list()
            for j, lt in enumerate(trainData):
                if j != i:
                    for point in lt:
                        tmp.append(point.point)
                        tmpLabels.append(labels[point.id])
            for point in predictData:
                pd.append(point.point)
            clf = SVC()
            X_transformed = pca.fit_transform(tmp)
            newdata_transformed = pca.transform(pd)
            # mean, stdDev = h.normalizeData(trainData)
            # h.normalizeEvaluationSet(predictData, mean, stdDev)
            clf.fit(np.array(X_transformed), np.array(tmpLabels))
            y_pred = clf.predict(np.array(newdata_transformed))
            k = 0
            for i in range(418, 796):
                finalAnswer[i].append(y_pred[k])
                k+=1
        for key in finalAnswer:
            matrix[key] = 0 if finalAnswer[key].count(0) > finalAnswer[key].count(1) else 1
        return matrix
    
    def knn2(self, predictData, trainData, labels):
        h = hp()
        matrix = defaultdict(list)
        # mean, stdDev = h.normalizeData(trainData)
        # h.normalizeEvaluationSet(predictData, mean, stdDev)
        finalAnswer = defaultdict(list)
        pd = list()
        pca = PCA(n_components=4)
        for point in predictData:
            pd.append(point.point)
        for i in range(len(trainData)):
            tmp = list()
            tmpLabels = list()
            pd = list()
            for j, lt in enumerate(trainData):
                if j != i:
                    for point in lt:
                        tmp.append(point.point)
                        tmpLabels.append(labels[point.id])
            for point in predictData:
                pd.append(point.point)
            knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
            X_transformed = pca.fit_transform(tmp)
            knn.fit(np.array(X_transformed), np.array(tmpLabels))
            newdata_transformed = pca.transform(pd)
            y_pred = knn.predict(np.array(newdata_transformed))
            k = 0
            for i in range(418, 796):
                finalAnswer[i].append(y_pred[k])
                k+=1
        for key in finalAnswer:
            matrix[key] = 0 if finalAnswer[key].count(0) > finalAnswer[key].count(1) else 1
        return matrix
    
    def gnb(self, predictData, trainData, labels):
        finalAnswer = defaultdict(list)
        pd = list()
        # clf = GaussianNB()
        clf = BernoulliNB(alpha=7.0)
        for point in predictData:
            pd.append(point.point)
        tmp = list()
        tmpLabels = list()
        for lt in trainData:
            for point in lt:
                tmp.append(point.point)
                tmpLabels.append(labels[point.id])
        # pca = PCA(n_components=10, svd_solver='full')
        # pca_matrix = pca.fit_transform(tmp)
        # pca1_matrix = pca.transform(pd)
        clf.fit(np.array(tmp), np.array(tmpLabels))
        y_pred = clf.predict(np.array(pd))
        k = 0
        for i in range(418, 796):
            finalAnswer[i]= int(y_pred[k])
            k+=1
        return finalAnswer
   
    def bayes_naive(self, predictData, trainData):
        h = hp()
        nb = bayes()
        matrix = defaultdict(list)
        pd = [pt for pt in predictData]
        for i in range(len(trainData)):
            tmp = None
            if predictData == None:
                predictData = trainData[i]
                tmp = [lt for j, lt in enumerate(trainData) if j != i]
            else:
                tmp = [lt for j, lt in enumerate(trainData) if j != i]
            td = h.convertToList(tmp)
            classPriorProbabilities = nb.findClassPriorProbability(td)
            classes = nb.segregateClasses(td)
            descriptorPosteriorProbabilites, occurences, means, stdDev = nb.findDescriptorPosteriorProbabilites(classes, td)
            nb.classify(predictData, classPriorProbabilities, descriptorPosteriorProbabilites, occurences, means, stdDev)
            for point in predictData:
                matrix[point.id].append(point.label)
            predictData = [pt for pt in pd]
        labels = defaultdict(list)
        for key in matrix:
            labels[key] = 0 if matrix[key].count(0) > matrix[key].count(1) else 1
        for point in predictData:
            point.label = labels[point.id]
        return predictData

    def ensemble_learning(self, predictData, trainData, labels):
        tmp = list()
        tmpLabels = list()
        pd = []
        estimators = []
        mean, stdDev = h.normalizeData(trainData)
        h.normalizeEvaluationSet(predictData, mean, stdDev)
        # pca = PCA(n_components=25)
        finalAnswer = dict()
        for lt in trainData:
            for point in lt:
                tmp.append(point.point)
                tmpLabels.append(labels[point.id])
        for point in predictData:
            pd.append(point.point)
        model1 = LogisticRegression()
        estimators.append(('logistic', model1))
        # model2 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        model4 = BernoulliNB()
        # estimators.append(('cart', model2))
        model3 = SVC()
        estimators.append(('svm', model3))
        estimators.append(('gnb', model4))
        ensemble = VotingClassifier(estimators)
        ensemble = ensemble.fit(tmp,tmpLabels)
        y_pred = ensemble.predict(pd)
        k = 0
        for i in range(418, 796):
            finalAnswer[i] = y_pred[k]
            k+=1
        return finalAnswer

    def adaboost(self, predictData, trainData, lables):
        tmp = list()
        tmpLabels = list()
        pd = []
        estimators = []
        finalAnswer = dict()
        for lt in trainData:
            for point in lt:
                tmp.append(point.point)
                tmpLabels.append(labels[point.id])
        tmp = np.array(tmp)
        tmpLabels = np.array(tmpLabels)
        for point in predictData:
            pd.append(point.point)
        classifier = AdaBoostClassifier(BernoulliNB(), n_estimators=300, random_state=0)
        classifier.fit(tmp, tmpLabels)
        predictions = classifier.predict(pd)
        finalAnswer = defaultdict()
        k = 0
        for i in range(418, 796):
            finalAnswer[i] = predictions[k]
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
    # h.calculateMetrics(accuracy, precision, recall, f_score)
    # m.bayes_naive(predictData, trainData)
    # fa = m.logisticRegression(predictData, trainData, labels)
    # fa = m.svm(predictData, trainData, labels)
    # fa = m.knn2(predictData, trainData, labels)
    fa = m.gnb(predictData, trainData, labels)
    # fa = m.ensemble_learning(predictData, trainData, labels)
    # fa = m.adaboost(predictData, trainData, labels)
    c.writeToCSV(fa)
    # c.writeToCSV(predictData)
    