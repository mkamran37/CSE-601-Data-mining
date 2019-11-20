from helpers import helpers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import xgboost
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def readTrainData():

    fileName = input("enter file name (without extension): ")
    filePath = "CSE-601/project3/Data/"+fileName+".csv"
    data = np.genfromtxt(filePath, dtype=None, delimiter=",", encoding=None)
    data = pd.DataFrame(data)
    data.drop(data.columns[0], axis=1, inplace=True)

    fileName = input("enter file name (without extension): ")
    filePath = "CSE-601/project3/Data/"+fileName+".csv"
    labels = np.genfromtxt(filePath, dtype=None, delimiter=",", encoding=None)
    labels = pd.DataFrame(labels)
    labels.drop(labels.index[0], axis=0, inplace=True)
    labels.drop(labels.columns[0], axis=1, inplace=True)
    labels = labels.reset_index(drop=True)

    return data, labels

def splitData(data, labels):

    train_features, test_features, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 5)
    test_labels = test_labels.ravel()
    train_labels = train_labels.ravel()
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    return train_features, test_features, train_labels, test_labels

def calMetrics(test_labels, pred_labels):

    print("Accuracy:", metrics.accuracy_score(test_labels, pred_labels))
    print("Precision:", metrics.precision_score(test_labels, pred_labels, pos_label='1'))
    print("Recall:", metrics.recall_score(test_labels, pred_labels, pos_label='1'))
    print("F-Measure:", metrics.f1_score(test_labels, pred_labels, pos_label='1'))
    print("F Beta Score:", metrics.fbeta_score(test_labels, pred_labels, beta=0.5, pos_label='1'))
    # print("MSE:", metrics.mean_squared_error(test_labels, pred_labels))

def readTestData():

    fileName = input("enter test file name (without extension): ")
    filePath = "CSE-601/project3/Data/"+fileName+".csv"
    test_features = np.genfromtxt(filePath, dtype=None, delimiter=",", encoding=None)
    test_features = pd.DataFrame(test_features)
    test_features.set_index(['f0'], inplace=True)
    return test_features

def writeToFile(test_features, pred_labels):

    f = open('CSE-601/project3/Data/output8.csv', 'w')
    for y, i in zip(pred_labels, test_features.index.values):
        f.write(str(i))
        f.write(',')
        f.write(str(y))
        f.write('\n')
    f.close()

def predict(clf, test_features):

    return clf.predict(test_features)

def rf(data, labels, test_features=None):

    from random_forest import randomForest
    from helpers import helpers as hp
    from decision_tree import decisionTree
    h = hp()
    rf = randomForest()
    dt = decisionTree()

    data = pd.concat([data, labels], axis=1)
    print(data)

    accuracy = []
    precision = []
    recall = []
    f_score = []
    models = []
    fb_score = []

    foldSize = int(data.shape[0] / 5)
    for i in range(5):
        print("Running iteration " + str(i+1) + " of k cross validation")
        testData = data.loc[foldSize*i:foldSize*(i+1)-1]
        trainData = data.loc[:foldSize*i-1].append(data.loc[foldSize*(i+1):])
        forest = rf.forest(trainData)
        target = testData.iloc[:,-1].values.tolist()
        predicted = rf.predictForest(testData.iloc[:, :-1], forest)
        models.append(forest)
        calMetrics(target, predicted)
        # truePositives, trueNegatives, falsePositives, falseNegatives = h.findParameters(predicted, target)
        # print(truePositives, trueNegatives, falsePositives, falseNegatives)
        # accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
        # tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
        # tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
        # precision.append(tmpPrecision)
        # recall.append(tmpRecall)
        # tm_fscore = h.findFMeasure(tmpPrecision, tmpRecall)
        # print(tm_fscore)
        # f_score.append(tm_fscore)
    
    h.calculateMetrics(accuracy, precision, recall, f_score)

    # for i in range(3):
    #     print("Running iteration " + str(i+1) + " of k cross validation")
    #     testData = data.loc[foldSize*i:foldSize*(i+1)-1]
    #     trainData = data.loc[:foldSize*i-1].append(data.loc[foldSize*(i+1):])
    #     root = dt.decision(trainData)
    #     target = testData.iloc[:,-1].values.tolist()
    #     predicted = dt.predictData(testData.iloc[:, :-1], root)
    #     models.append(root)
    #     truePositives, trueNegatives, falsePositives, falseNegatives = h.findParameters(predicted, target)
    #     accuracy.append(h.findAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives))
    #     tmpPrecision = h.findPrecision(truePositives, trueNegatives, falsePositives, falseNegatives)
    #     tmpRecall = h.findRecall(truePositives, trueNegatives, falsePositives, falseNegatives)
    #     precision.append(tmpPrecision)
    #     recall.append(tmpRecall)
    #     f_score.append(h.findFMeasure(tmpPrecision, tmpRecall))
    
    # print(accuracy, precision, recall, f_score)
    # h.calculateMetrics(accuracy, precision, recall, f_score)

    # ind = f_score.index(min(f_score))
    # print(f_score[ind])
    # pred = rf.predictForest(test_features, models[ind])
    # print(pred)
    predicted = pd.DataFrame()
    for root in models:
        pred = dt.predictData(test_features, root)
        predicted = pd.concat([predicted, pd.DataFrame(pred)], axis=1)

    print(predicted)

    p = pd.DataFrame()

    p = []
    for idx, row in predicted.iterrows():
        p.append(row.value_counts().index.tolist()[0])

    print(p)

    return p
    # print([max(set(pred), key=pred.count) for pred in predicted])

def randomForest(train_features, train_labels):

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_features, train_labels)
    return clf

def adaBoost(train_features, train_labels):

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=200)
    clf.fit(train_features, train_labels)

    return clf

def xgb(train_features, train_labels):

    clf = xgboost.XGBClassifier(random_state=1, learning_rate=0.01, n_estimators=200, max_depth=5)
    clf.fit(train_features, train_labels)
    return clf

def nn(train_features, train_labels):
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(train_features, train_labels)
    return mlp

def lr(train_features, train_labels):
    clf = LinearRegression()
    clf.fit(train_features, train_labels)
    return clf

if __name__ == "__main__":

    data, labels = readTrainData()
    # print(labels.iloc[:,0].value_counts())
    # print(zeros, ones)
    # trainData = data.iloc[:int(data.shape[0]*0.8)]
    # trainLabels = labels.iloc[:int(labels.shape[0]*0.8)]
    # testData = data.iloc[int(data.shape[0]*0.8):]
    # testLabels = labels.iloc[int(labels.shape[0]*0.8):]
    # minmaxScaler = preprocessing.MinMaxScaler()
    # scaledData = minmaxScaler.fit_transform(data)
    # StandardScaler().fit_transform(data)
    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(data.values)
    # principalDf = pd.DataFrame(data = principalComponents
    #          , columns = ['principal component 1', 'principal component 2'])
    # finalDf = pd.concat([principalDf, labels], axis = 1)
    # print(finalDf.head())
    
    # zeros = finalDf[finalDf[1] == '0']
    # ones = finalDf[finalDf[1] == '1']

    # newDf = pd.concat([zeros, ones])
    # print(newDf.head())
    # newdf = newDf.sample(frac=1, random_state=42)

    # data = newdf.drop([1], axis=1)
    # labels = pd.DataFrame(newdf[1])

    # print(pd.DataFrame(data).shape)
    # exit()
    # exit()
    # labels = np.array(labels)
    data = np.array(data)
    train_features, test_features, train_labels, test_labels = splitData(data, np.array(labels))
    train_features = normalize(train_features)
    test_features=normalize(test_features)
    neighbours = np.arange(1,25)
    train_accuracy =np.empty(len(neighbours))
    test_accuracy = np.empty(len(neighbours))
    for i,k in enumerate(neighbours):
        # knn=KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree",n_jobs=-1)
        # knn.fit(train_features,train_labels.ravel())
        # train_accuracy[i] = knn.score(train_features, train_labels.ravel())
        # test_accuracy[i] = knn.score(test_features, test_labels.ravel())
        clf = xgb(train_features, train_labels)
        


    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbours, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()

    idx = np.where(test_accuracy == max(test_accuracy))
    x = neighbours[idx]

    knn=KNeighborsClassifier(n_neighbors=x[0],algorithm="kd_tree",n_jobs=-1)
    knn.fit(train_features,train_labels.ravel())
    pred = knn.predict(test_features)
    # calMetrics(test_labels, pred)
    # exit()
    # rf(data, labels)
    # clf = adaBoost(train_features, train_labels)
    # clf = nn(train_features, train_labels)
    # clf = lr(train_features, train_labels)

    # xgboost.plot_importance(clf)
    # plt.show()  

    # pred = predict(clf, test_features)
    # p = []
    # for pr in pred:
    #     if pr <= 0: p.append('0')
    #     else: p.append('1')
    # print(p)
    # calMetrics(test_labels, p)

    # pred = predict(clf, train_features)
    # calMetrics(train_labels, pred)

    test_features = normalize(readTestData())
    pred = knn.predict(np.array(test_features))
    print(pred)
    # print(train_features, test_features)
    # exit()

    # pred = predict(clf, np.array(testData))
    # calMetrics(np.array(testLabels), pred)

    writeToFile(test_features, pred)




