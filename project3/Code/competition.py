from helpers import helpers as hp
from knn import knn
from naive_bayes import bayes
from sklearn import preprocessing
from competition_helpers import competition as cp
import numpy as np
from collections import defaultdict

class main():
    def knn(self, predictData = None, trainData = None, kCrossValidation = 10):
        h = hp()
        matrix = defaultdict(list)
        mean, stdDev = h.normalizeData(trainData)
        h.normalizeEvaluationSet(predictData, mean, stdDev)
        pd = [p for p in predictData]
        for i in range(len(trainData)):
            tmp = [lt for j, lt in enumerate(trainData) if j != i]
            td = h.convertToList(tmp)
            knn().classify(td, predictData)
            for point in predictData:
                matrix[point.id].append(point.label)
            predictData = [p for p in pd]
        labels = defaultdict(list)
        for key in matrix:
            labels[key] = 0 if matrix[key].count(0) > matrix[key].count(1) else 1
        for point in predictData:
            point.label = labels[point.id]
        return predictData

if __name__ == "__main__":
    m = main()
    h = hp()
    c = cp()
    print("Enter train File name")
    trainData = c.get_file_competition(h.get_fileName())
    print("Enter the labels file")
    c.assign_labels(trainData, c.read_labels("../Data/"+h.get_fileName()+".csv"))
    print("Enter the data to be predicted")
    name = h.get_fileName()
    predictData = c.get_file_competition(name, fileType='predictData')
    m.knn(predictData, trainData)
    c.writeToCSV(predictData)
    