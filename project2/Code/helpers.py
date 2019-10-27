from collections import defaultdict
from point import Point
import numpy as np
import pandas as pd

class helpers:
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
    
    def create_pd(self, dataset):
        geneID = [data.id for data in dataset]
        geneCluster = [data.cluster for data in dataset]
        points = np.array(geneCluster)
        ids = np.array(geneID)
        predicted = pd.DataFrame(data=points, index=ids, columns=["clusterNum"])
        return ids, predicted