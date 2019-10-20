import numpy as np
from scipy.spatial import distance
import random
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from math import e

class Point:
    def __init__(self, point=None, id=-1, cluster=-1):
        self.point = point
        self.cluster = cluster
        self.id = id


class Spectral:
    def __init__(self):
        filename = input("enter file name (without extension)")
        dataset = self.read_data("../Data/"+filename+".txt")
        W = self.computeSimilarityMatrix(dataset)
        D = self.computeDegreeMatrix(W)
        L = self.computeLaplaciaMatrix(D, W)
        eVal, eVector = self.findEigens(L)
        embeddedSpace = self.sort(eVal, eVector)
        # embeddedSpace = self.extract(eVector)
        data = self.simulateDataset(embeddedSpace)
        centroids = np.array(self.initializeCentroids(data))
        clusters = self.assignClusters(data, centroids)
        result = self.sort_result(data)
        # print(result)
        self.pca(dataset, result)

    def simulateDataset(self, dataset):
        '''
            input: dataset - top k eigen vectors
            output: data - a dictionary containing key as gene ID and value as the corresponding Point object
        '''
        data = dict()
        for i in range(len(dataset)):
            pt = Point()
            pt.id = i
            pt.point = dataset[i]
            data[i] = pt
        return data

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

    def computeSimilarityMatrix(self, dataset, sigma=5):
        '''
            input:  dataset - a list of Point objects
                    sigma - parameter of calculating gaussian kernel
            output: similarityMatrix - a NxN matrix, where N is the size of dataset,                                    consisting of gaussian weights between genes
        '''
        similarityMatrix = [[0 for x in range(len(dataset))] for y in range(len(dataset))]
        for point in dataset:
            for p in dataset:
                exp = np.power(np.linalg.norm(point.point - p.point), 2)
                s = sigma**2
                similarityMatrix[point.id-1][p.id-1] = np.power(e, -1*exp/s)
                # similarityMatrix[point.id-1][p.id-1] = distance.euclidean(point.point, p.point)
        return similarityMatrix

    def computeDegreeMatrix(self, W):
        '''
        input:  W - similarityMatrix
        output: D - a NxN matrix, where N is the size of similarityMatrix, defining the degree.
        '''
        res = np.sum(W,axis=1).tolist()
        D = [[0 for _ in range(len(W))] for _ in range(len(W))]
        for i in range(len(D)):
            D[i][i] = res[i]
        return D

    def computeLaplaciaMatrix(self, D, W):
        '''
        input:  D, W - NxN matrices
        output: L   -  NxN laplacian matrix
        '''
        a = np.array(D)
        b = np.array(W)
        return a-b

    def findEigens(self, L):
        '''
        input:  L - Laplacian Matrix
        output: eigenValues, eigenVectors corresponding to L
        '''
        return np.linalg.eig(L)

    def sort(self, eigenValues, eigenVectors, k = 5):
        '''
        input:  eigenValues, eigenVectors
        output: eigen vectors corresponding to the sorted eigen values in ascending order
        '''
        idx = eigenValues.argsort()[:k]
        # eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        # print(eigenVectors, eigenValues)
        return eigenVectors

    def extract(self, eVector, k = 5):
        '''
        input:  eVector - eigen vector sorted in ascending order according to the eigen values
                k - number of desired clusters
        output: top K smallest eigen vectors
        '''
        # eVector = eVector[:k]
        return eVector[:,:k]

    def initializeCentroids(self, dataset, k=5):
        '''
        input:  dataset - a dictionary containing key as gene ID and value as the corresponding Point object
                k - number of desired clusters
        output: centroids - k number of initial random centroids
        '''
        centroids = list()
        while k > 0:
            idx = int(input("enter id"))
            centroids.append(dataset[idx+1].point)
            k-=1
        return centroids

    def assignClusters(self, dataset, centroids, iterations = 200):
        # prevCentroids = np.empty_like(centroids)
        clusters = defaultdict(list)
        j = 0
        while j < iterations:
            # prevCentroids = centroids
            clusters = defaultdict(list)
            for i in range(len(dataset)):
                clusters = self.find_cluster(centroids, dataset[i], clusters)
            centroids = self.findClusterCentroid(centroids, clusters)
            j+=1
        return clusters

    def sort_result(self, datasets):
        cluster = defaultdict(list)
        dictlist = []
        for i in datasets.keys():
            cluster[int(datasets[i].id)].append(int(datasets[i].cluster))
        for key, value in cluster.items():
            temp = [key,value[0]]
            dictlist.append(temp)
        dictlist.sort(key= lambda x:x[0])
        return dictlist
        
    def find_cluster(self, centroids, gene, clusters):
        min_dist = float('inf')
        cluster = 0
        for i,centroid in enumerate(centroids):
            dist = np.linalg.norm(gene.point - centroid)
            if dist < min_dist:
                min_dist = dist
                cluster = i+1
        gene.cluster = int(cluster)
        clusters[cluster].append(gene.point)
        return clusters

    def findClusterCentroid(self, centroids, clusters):
        for i,key in enumerate(clusters):
            centroids[i] = np.array(clusters[key], dtype=np.float64).mean(axis=0)
        return centroids

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
    s = Spectral()