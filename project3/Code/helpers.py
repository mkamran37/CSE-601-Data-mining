import numpy as np
from point import point
import math
import pandas as pd
from collections import defaultdict

class helpers:
    def get_fileName(self):
        filename = input("Enter file name (without extension): ")
        return filename

    def get_file_bayes(self, filename, kCrossValidation = 10,  fileType='trainData'):
        if fileType == 'predictData':
            data = self.read_predictData_bayes("../Data/"+filename+".txt")
        else:
            data = self.read_data_bayes("../Data/"+filename+".txt", kCrossValidation)
        return data
        
    def get_file_bayes_demo(self, filename, fileType='trainData'):
        if fileType == 'predictData':
            data = self.read_predictData_bayesDemo("../Data/"+filename+".txt")
        else:
            data = self.read_data_bayes_demo("../Data/"+filename+".txt")
        return data

    def get_file(self, filename, kCrossValidation = 10,  fileType='trainData'):
        if fileType == 'predictData':
            trainData = self.read_predictData("../Data/"+filename+".txt")
        else:
            trainData = self.read_data("../Data/"+filename+".txt", kCrossValidation)
        return trainData
    
    def get_file_demo(self, filename,  fileType='trainData'):
        if fileType == 'predictData':
            data = self.read_predictData_demo("../Data/"+filename+".txt")
        else:
            data = self.read_data_demo("../Data/"+filename+".txt")
        return data

    def read_predictData_demo(self, filepath):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        tmp = list()
        for i in range(file.shape[0]):
            data = point()
            temp = list()
            for j in range(file.shape[1]):
                if j == file.shape[1]-1:
                    data.label = int(file[i][j])
                    data.groundTruth = data.label
                else:
                    n = float(file[i][j])
                    temp.append(n)
            data.point = np.array(temp)
            tmp.append(data)
        return tmp

    def read_predictData(self, filepath):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        start = 0.0
        nominal_to_number = dict()
        tmp = list()
        for i in range(file.shape[0]):
            data = point()
            temp = list()
            for j in range(file.shape[1]):
                if j == file.shape[1]-1:
                    data.label = int(file[i][j])
                    data.groundTruth = data.label
                else:
                    try:
                        n = float(file[i][j])
                        temp.append(n)
                    except:
                        if file[i][j] not in nominal_to_number:
                            nominal_to_number[file[i][j]] = start
                            start+=1.0
                        temp.append(nominal_to_number[file[i][j]])
            data.point = np.array(temp)
            tmp.append(data)
        return tmp

    def read_predictData_bayes(self, filepath):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        tmp = list()
        for i in range(file.shape[0]):
            data = point()
            temp = list()
            catData = list()
            for j in range(file.shape[1]):
                if j == file.shape[1]-1:
                    data.label = int(file[i][j])
                    data.groundTruth = data.label
                else:
                    try:
                        n = float(file[i][j])
                        temp.append(n)
                    except:
                        catData.append(file[i][j])
            data.point = np.array(temp)
            data.categoricalData = np.array(catData)
            tmp.append(data)
        return tmp

    def read_predictData_bayesDemo(self,filepath):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        tmp = list()
        catData = list()
        data = point()
        for i in range(file.shape[0]):
            catData.append(file[i])
        data.categoricalData = np.array(catData)
        tmp.append(data)
        return tmp
    
    def read_data_bayes_demo(self, filepath):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        trainData = list()
        for i in range(file.shape[0]):
            data = point()
            catData = list()
            for j in range(file.shape[1]):
                if j == file.shape[1]-1:
                    data.label = int(file[i][j])
                    data.groundTruth = data.label
                else:
                    catData.append(file[i][j])
            data.categoricalData = np.array(catData)
            trainData.append(data)
        return trainData

    def read_data_bayes(self, filepath, kCrossValidation):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        trainData = list()
        k = 0
        counter = 0
        maxsize = math.ceil(file.shape[0]/kCrossValidation)+1
        # For d4 take ceil else take floor
        # maxsize = math.floor(file.shape[0]/kCrossValidation)
        while k < kCrossValidation:
            extent = counter
            tmp = list()            
            for i in range(extent, min(extent+maxsize, file.shape[0])):
                data = point()
                temp = list()
                catData = list()
                for j in range(file.shape[1]):
                    if j == file.shape[1]-1:
                        data.label = int(file[i][j])
                        data.groundTruth = data.label
                    else:
                        try:
                            n = float(file[i][j])
                            temp.append(n)
                        except:
                            catData.append(file[i][j])
                data.point = np.array(temp)
                data.categoricalData = np.array(catData)
                tmp.append(data)
                counter+=1
            if len(tmp) != 0:
                trainData.append(tmp)
            k+=1
        return trainData
   
    def read_data_demo(self, filepath):
        '''
            :type   filepath
            :rtype: trainData - a list of Point objects with known labels used to train the model
            :rtype: predictData - a list of Point objects with unknown labels
        '''
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        trainData = list()
        for i in range(file.shape[0]):
            data = point()
            temp = list()
            for j in range(file.shape[1]):
                if j == file.shape[1]-1:
                    data.label = int(file[i][j])
                    data.groundTruth = data.label
                else:
                    n = float(file[i][j])
                    temp.append(n)
            data.point = np.array(temp)
            trainData.append(data)
        return trainData

    def read_data(self, filepath, kCrossValidation = 10):
        '''
            :type   filepath
            :rtype: trainData - a list of Point objects with known labels used to train the model
            :rtype: predictData - a list of Point objects with unknown labels
        '''
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        trainData = list()
        k = 0
        counter = 0
        start = 0.0
        nominal_to_number = defaultdict(int)
        maxsize = math.ceil(file.shape[0]/kCrossValidation)+1
        # maxsize = math.floor(file.shape[0]/kCrossValidation)
        while k < kCrossValidation:
            extent = counter
            tmp = list()
            for i in range(extent, min(extent+maxsize, file.shape[0])):
                data = point()
                temp = list()
                for j in range(file.shape[1]):
                    if j == file.shape[1]-1:
                        data.label = int(file[i][j])
                        data.groundTruth = data.label
                    else:
                        try:
                            n = float(file[i][j])
                            temp.append(n)
                        except:
                            if file[i][j] not in nominal_to_number:
                                nominal_to_number[file[i][j]]=start
                                start+=1.0
                            temp.append(nominal_to_number[file[i][j]])
                data.point = np.array(temp)
                tmp.append(data)
                counter+=1
            if len(tmp) != 0:
                trainData.append(tmp)
            k+=1
        return trainData
    
    def convertToList(self, td):
        result = list()
        for lt in td:
            for pt in lt:
                result.append(pt)
        return result
    
    def findParams(self, predictData):
        truePositives, trueNegatives, falsePositives, falseNegatives = 0,0,0,0
        for pt in predictData:
            if pt.label == 1 and pt.groundTruth == 1:
                truePositives+=1
            elif pt.label == 1 and pt.groundTruth == 0:
                falsePositives+=1
            elif pt.label == 0 and pt.groundTruth == 1:
                falseNegatives+=1
            else:
                trueNegatives+=1
        return truePositives, trueNegatives, falsePositives, falseNegatives

    def findAccuracy(self, truePositives, trueNegatives, falsePositives, falseNegatives):
        return (truePositives+trueNegatives)/(truePositives+trueNegatives+falsePositives+falseNegatives)
    
    def findPrecision(self, truePositives, trueNegatives, falsePositives, falseNegatives):
        try:
            return (truePositives)/(truePositives+falsePositives)
        except:
            return 0
    
    def findRecall(self, truePositives, trueNegatives, falsePositives, falseNegatives):
        try:
            return (truePositives)/(truePositives+falseNegatives)
        except:
            return 0
    
    def findFMeasure(self, precision, recall):
        try:
            return (2*precision*recall)/(precision+recall)
        except:
            return 0
    
    def normalizeData(self, data):
        tmp = list()
        for lt in data:
            for point in lt:
                tmp.append(point.point)
        tmp = np.array(tmp)
        mean = tmp.mean(axis=0)
        stdDev = tmp.std(axis=0)
        for lst in data:
            for point in lst:
                pt = list()
                for i in range(len(point.point)):
                    pt.append((point.point[i] - mean[i])/(stdDev[i]))
                point.point = np.array(pt)
        return mean, stdDev
    
    def normalizeEvaluationSet(self, data, mean, stdDev):
        for point in data:
            pt = list()
            for i in range(len(point.point)):
                pt.append((point.point[i] - mean[i])/(stdDev[i]))
            point.point = np.array(pt)
    
    def standardizeBayes(self, data):
        tmp = list()
        for lt in data:
            tmp.append(lt.point)
        tmp = np.array(tmp)
        mean = tmp.mean(axis=0)
        stdDev = tmp.std(axis=0)
        return mean, stdDev
    
    def calculateMetrics(self, accuracy, precision, recall, f_score):
        averageAccuracy = sum(accuracy)/len(accuracy)
        averagePrecision = sum(precision)/len(precision)
        averageRecall = sum(recall)/len(recall)
        averageFscore = sum(f_score)/len(f_score)
        print("ACCURACY = {}".format(averageAccuracy))
        print("PRECISION = {}".format(averagePrecision))
        print("RECALL = {}".format(averageRecall))
        print("F MEASURE = {}".format(averageFscore))

    def calculateMetricsDemo(self, accuracy, precision, recall, f_score):
        print("ACCURACY = {}".format(accuracy))
        print("PRECISION = {}".format(precision))
        print("RECALL = {}".format(recall))
        print("F MEASURE = {}".format(f_score))

    def readData(self, filePath):
        '''
            Read input data for decision tree and random forest classifier
            input: filepath
            output: Data Points- a pandas dataframe of input data
                    Labels - a pandas dataframe of labels for each data point
        '''
        data = np.genfromtxt(filePath, dtype=None, delimiter="\t", encoding=None)
        dataDf = pd.DataFrame(data)
        labels = dataDf.iloc[:,-1]
        return dataDf.iloc[:,:-1], dataDf.iloc[:,-1]

    def oneHotEncoding(self, data, labels):
        '''
            One Hot Encode the input data file and then concat the labels to return a single dataframe
            input:  data - pandas dataframe of input data 
                    labels - pandas dataframe of labels associated with input data points
            output: returns a dataframe with one hot encoding and joining the labels to the data points
        '''
        for colName, colData in data.iteritems():
            if colData.dtype == np.object:
                data = pd.concat([data, pd.get_dummies(colData, prefix=colName)], axis=1)
                data.drop([colName], axis=1, inplace=True)

        return pd.concat([data, labels], axis=1)

    def findParameters(self, predicted, target, tp=1, tn=0):
        truePositives, trueNegatives, falsePositives, falseNegatives = 0,0,0,0
        for p, t in zip(predicted, target):
            if p == tp and t == tp:
                truePositives+=1
            elif p == tp and t == tn:
                falsePositives+=1
            elif p == tn and t == tp:
                falseNegatives+=1
            else:
                trueNegatives+=1
        return truePositives, trueNegatives, falsePositives, falseNegatives
