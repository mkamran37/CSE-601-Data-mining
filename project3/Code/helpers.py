import numpy as np
from point import point
import math
import pandas as pd

class helpers:
    def get_fileName(self):
        filename = input("enter file name (without extension): ")
        return filename

    def get_file(self, filename, kCrossValidation = 10, fileType = 'trainData'):
        trainData = self.read_data("../Data/"+filename+".txt", kCrossValidation, fileType)
        # dataset = self.read_data("CSE-601/project2/Data/"+filename+".txt")
        if fileType == 'predictData':
            trainData = self.read_predictData("../Data/"+filename+".txt")
        return trainData
    
    def read_predictData(self, filepath):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        trainData = list()
        k = 0
        counter = 0
        start = 0.0
        nominal_to_number = dict()
        tmp = list()
        for i in range(file.shape[0]):
            data = point()
            temp = list()
            for j in range(file.shape[1]):
                if j == file.shape[1]-1:
                    # data.label = int(file[i][j])
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

    def read_data(self, filepath, kCrossValidation, filetype):
        '''
            input: filepath
            output: trainData - a list of Point objects with known labels used to train the model
                    predictData - a list of Point objects with unknown labels
        '''
        file = np.genfromtxt(filepath, dtype='unicode', delimiter="\t")
        trainData = list()
        k = 0
        counter = 0
        start = 0.0
        nominal_to_number = dict()
        maxsize = math.ceil(file.shape[0]/kCrossValidation)+1
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
                                nominal_to_number[file[i][j]] = start
                                start+=1.0
                            temp.append(nominal_to_number[file[i][j]])
                data.point = np.array(temp)
                tmp.append(data)
                counter+=1
            if len(tmp) != 0:
                trainData.append(tmp)
            if filetype == 'predictData':
                return tmp
            k+=1
        return trainData
    
    def convertToList(self, td):
        result = list()
        for lt in td:
            for pt in lt:
                result.append(pt)
        return result
    
    def findParams(self, predictData, tp = 1, tn = 0):
        truePositives, trueNegatives, falsePositives, falseNegatives = 0,0,0,0
        for pt in predictData:
            if pt.label == tp and pt.groundTruth == tp:
                truePositives+=1
            elif pt.label == tp and pt.groundTruth == tn:
                falsePositives+=1
            elif pt.label == tn and pt.groundTruth == tp:
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
        mean = dict()
        stdDeviation = dict()
        for lst in data:
            for point in lst:
                for i in range(len(point.point)):
                    if i not in mean:
                        mean[i] = self.findMean(i, data)
                    if i not in stdDeviation:
                        stdDeviation[i] = self.findstdDeviation(i, data, mean)
                    point.point[i] = (point.point[i] - mean[i])/(stdDeviation[i])
    
    def normalizeEvaluationSet(self, data):
        mean = dict()
        stdDeviation = dict()
        for point in data:
            for i in range(len(point.point)):
                if i not in mean:
                    mean[i] = self.findMeanES(i, data)
                if i not in stdDeviation:
                    stdDeviation[i] = self.findStdDeviationES(i, data, mean)
                if stdDeviation[i] == 0.0:
                    point.point[i] = (point.point[i] - mean[i])
                else:
                    point.point[i] = (point.point[i] - mean[i])/(stdDeviation[i])
    
    def findMeanES(self, index, data):
        sumMean = 0
        for point in data:
            sumMean += point.point[index]
        return sumMean/len(data)
    
    def findStdDeviationES(self, index, data, mean):
        stdDev = 0
        for point in data:
            stdDev += ((point.point[index] - mean[index])**2)
        if len(data) > 1:
            return (stdDev/(len(data) - 1))**0.5
        else:
            return (stdDev/(len(data)))**0.5
   
    def findMean(self, index, data):
        sumMean = 0
        for lst in data:
            for point in lst:
                sumMean += point.point[index]
        return sumMean/len(data)
    
    def findstdDeviation(self, index, data, mean):
        stdDev = 0
        for lst in data:
            for point in lst:
                stdDev = stdDev + ((point.point[index] - mean[index])**2)
        if len(data) > 1:
            return (stdDev/(len(data) - 1))**0.5
        else:
            return (stdDev/(len(data)))**0.5

    def calculateMetrics(self, accuracy, precision, recall, f_score):
        averageAccuracy = sum(accuracy)/len(accuracy)
        averagePrecision = sum(precision)/len(precision)
        averageRecall = sum(recall)/len(recall)
        averageFscore = sum(f_score)/len(f_score)
        print("ACCURACY = {}%".format(averageAccuracy*100))
        print("PRECISION = {}%".format(averagePrecision*100))
        print("RECALL = {}%".format(averageRecall*100))
        print("F MEASURE = {}%".format(averageFscore*100))

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

    def findParameters(self, predicted, target, tp='1', tn='0'):
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
