from collections import defaultdict
from point import Point
import numpy as np
import pandas as pd
from External_Index import externalIndex

class helpers:
    def get_file(self):
        filename = input("enter file name (without extension): ")
        dataset = self.read_data("../Data/"+filename+".txt")
        # dataset = self.read_data("CSE-601/project2/Data/"+filename+".txt")
        return dataset, filename

    def sort_result(self, datasets):
        cluster = defaultdict(list)
        dictlist = []
        for dataset in datasets:
            cluster[int(dataset.id)].append(int(dataset.cluster))
        for key, value in cluster.items():
            temp = [key,value[0]]
            dictlist.append(temp)
        dictlist.sort(key= lambda x:x[0])
        return dictlist

    def read_data(self, filepath):
        '''
            input: filepath
            output: dataset - a list of Point objects
        '''
        data = np.genfromtxt(filepath, dtype='double', delimiter="\t")
        dataset = list()
        for i in range(data.shape[0]):
            tmp = list()
            gene = Point()
            for j in range(data.shape[1]):
                if j == 0:
                    gene.id = int(data[i][0])
                elif j == 1:
                    continue
                else:
                    tmp.append(data[i][j])
            gene.point = np.array(tmp)
            dataset.append(gene)
        return dataset
    
    def create_pd(self, datasett):
        dataID = [data.id for data in datasett]
        dataCluster = [data.cluster for data in datasett]
        dataset = [data.point for data in datasett]
        points = np.array(dataCluster)
        ids = np.array(dataID)
        predicted = pd.DataFrame(data=points, index=ids, columns=["Cluster"])
        return dataset, ids, predicted
    
    def calculateCoeff(self, predicted, filename, ids):
        # ids, predicted = self.create_pd(self, dataset)
        # groundTruth = np.genfromtxt("CSE-601/project2/Data/"+filename+".txt", delimiter="\t", dtype=str, usecols=1)
        groundTruth = np.genfromtxt("../Data/"+filename+".txt", delimiter="\t", dtype=str, usecols=1)
        coeff = externalIndex(predicted, groundTruth, ids)
        rand, jaccard = coeff.getExternalIndex()
        print("RAND COEFFICIENT: {}".format(rand))
        print("JACCARD COEFFICIENT: {}".format(jaccard))