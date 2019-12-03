import numpy as np
from point import point
import math
import csv
from collections import defaultdict

class competition:
    
    def get_file_competition(self, filename, kCrossValidation = 9, fileType='trainData'):
        if fileType == 'predictData':
            trainData = self.read_predictData("../Data/"+filename+".csv")
        else:
            trainData = self.read_data_competition("../Data/"+filename+".csv", kCrossValidation)
        return trainData
        
    def read_data_competition(self, filepath, kCrossValidation = 10):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter=",")
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
                    if j == 0:
                        data.id = int(file[i][j])
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
            k+=1
        return trainData

    def read_labels(self, filepath):
        # file = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=int)
        # labels = dict()
        # for row in file:
        #     labels[row[0]] = row[1]
        # return labels
        file = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=int)
        labels = defaultdict(int)
        for row in file:
            labels[row[0]] = row[1]
        return labels

    def assign_labels(self, file, labels):
        for row in file:
            for point in row:
                point.groundTruth = labels[point.id]
    
    def read_predictData(self, filepath):
        file = np.genfromtxt(filepath, dtype='unicode', delimiter=",")
        start = 0.0
        nominal_to_number = dict()
        tmp = list()
        for i in range(file.shape[0]):
            data = point()
            temp = list()
            for j in range(file.shape[1]):
                if j == 0:
                    data.id = int(file[i][j])
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
    
    def writeToCSV(self, predictData):
        f = open('../KNN.csv','w')
        for key in predictData:
            f.write(str(key))
            f.write(',')
            f.write(str(int(predictData[key])))
            f.write('\n')
        f.close()