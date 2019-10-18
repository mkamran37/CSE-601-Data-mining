from collections import defaultdict
import numpy as np
from scipy.spatial import distance
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


class Point:
    def __init__(self, point=None, id=-1, cluster=-1):
        self.point = point
        self.cluster = cluster
        self.id = id
class DensityBasedClustering:
    def __init__(self):
        filename = input("enter file name (without extension)")
        dataset = self.read_data("../Data/"+filename+".txt")
        distance = self.findDistanceMatrix(dataset)
        self.dbScan(dataset, distance=distance)
        result = self.sort_result(dataset)
        self.pca(dataset, result)

        
    def findDistanceMatrix(self, dataset):
        print(len(dataset))
        distanceMatrix = [[0 for x in range(len(dataset))] for y in range(len(dataset))]
        for point in dataset:
            for p in dataset:
                distanceMatrix[point.id-1][p.id-1] = distance.euclidean(point.point, p.point)
        return distanceMatrix
    def read_data(self, filepath):
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
            gene.point = tmp
            dataset.append(gene)
        return dataset
    def dbScan(self, dataset, eps=3.5, minpts=4, distance=None, points=None):
        clusterNumber = 0
        cluster = defaultdict(list)
        visited = set()
        for point in dataset:
            if point not in visited:
                visited.add(point)
                neigbors = self.regionQuery(point, eps, distance, dataset)
                if len(neigbors) < minpts:
                    continue
                else:
                    clusterNumber+=1
                    self.expandCluster(point, neigbors, cluster, eps, minpts, clusterNumber, visited, distance, dataset)
        return cluster
    
    def expandCluster(self, point, neighbors, clusters, eps, minpts, clusterNumber, visited, distance, dataset):
        clusters[clusterNumber].append(point)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                newNeighbours = self.regionQuery(neighbor, eps, distance, dataset)
                if len(newNeighbours) >= minpts:
                    neighbors = newNeighbours.append(neighbors)
            if neighbor.cluster == -1:
                clusters[clusterNumber].append(neighbor)
                neighbor.cluster = clusterNumber
    
    def regionQuery(self, neighbor, eps, distance, points):
        result = list()
        for point in points:
            if distance[neighbor.id-1][point.id-1] < eps:
                result.append(point)
        return result

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


    def pca(self, datasett, result):
        pca = PCA(n_components=2, svd_solver='full')
        dataset = [data.point for data in datasett]
        pca.fit(dataset)
        r = np.array(result)
        pca_matrix = pca.transform(dataset)
        # print(pca_matrix.shape, r.shape)
        df = pd.DataFrame(data = np.concatenate((pca_matrix, r[:,1:2]), axis = 1), columns=['PC1','PC2','Cluster'])
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        plt.show()

if __name__ == "__main__":
    k = DensityBasedClustering()

