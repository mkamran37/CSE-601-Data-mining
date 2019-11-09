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
        return (truePositives)/(truePositives+falsePositives)
    
    def findRecall(self, truePositives, trueNegatives, falsePositives, falseNegatives):
        return (truePositives)/(truePositives+falseNegatives)
    
    def findFMeasure(self, precision, recall):
        return (2*precision*recall)/(precision+recall)