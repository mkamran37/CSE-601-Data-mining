import numpy as np
from point import point
import math
from collections import defaultdict

class helpers:
    def get_fileName(self):
        filename = input("enter file name (without extension): ")
        return filename

    def get_file_bayes(self, filename, kCrossValidation = 10,  fileType='trainData'):
        if fileType == 'predictData':
            trainData = self.read_predictData_bayes("../Data/"+filename+".txt")
        else:
            trainData = self.read_data_bayes("../Data/"+filename+".txt", kCrossValidation)
        return trainData

    def get_file(self, filename, kCrossValidation = 10,  fileType='trainData'):
        if fileType == 'predictData':
            trainData = self.read_predictData("../Data/"+filename+".txt")
        else:
            trainData = self.read_data("../Data/"+filename+".txt", kCrossValidation)
        return trainData
    
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

    def read_data(self, filepath, kCrossValidation):
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
        print("ACCURACY = {}%".format(averageAccuracy*100))
        print("PRECISION = {}%".format(averagePrecision*100))
        print("RECALL = {}%".format(averageRecall*100))
        print("F MEASURE = {}%".format(averageFscore*100))
