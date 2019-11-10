import numpy as np
from point import point
import math

class helpers:
    def get_fileName(self):
        filename = input("enter file name (without extension): ")
        return filename

    def get_file(self, filename, kCrossValidation = 10):
        trainData = self.read_data("../Data/"+filename+".txt", kCrossValidation)
        # dataset = self.read_data("CSE-601/project2/Data/"+filename+".txt")
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
        start = 0
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
                                start+=1
                            temp.append(nominal_to_number[file[i][j]])
                data.point = np.array(temp)
                tmp.append(data)
                counter+=1
            trainData.append(tmp)
            k+=1
        return trainData
    
    def convertToList(self, td):
        result = list()
        for lt in td:
            for pt in lt:
                result.append(pt)
        return result
    
    def findParams(self, predictData, tp, tn):
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
        return (stdDev/(len(data) - 1))**0.5
   
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
        return (stdDev/(len(data) - 1))**0.5

    def calculateMetrics(self, accuracy, precision, recall, f_score):
        averageAccuracy = sum(accuracy)/len(accuracy)
        averagePrecision = sum(precision)/len(precision)
        averageRecall = sum(recall)/len(recall)
        averageFscore = sum(f_score)/len(f_score)
        print("ACCURACY = {}%".format(averageAccuracy*100))
        print("PRECISION = {}%".format(averagePrecision*100))
        print("RECALL = {}%".format(averageRecall*100))
        print("F MEASURE = {}%".format(averageFscore*100))
