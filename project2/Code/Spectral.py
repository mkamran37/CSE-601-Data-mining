import numpy as np
from collections import defaultdict
from math import e
from point import Point
from visualization import visualization as vs
from helpers import helpers as hp
from K_means import k_means as km

class Spectral:
    def __init__(self):
        filename = input("enter file name (without extension)")
        dataset = hp.read_data(self, "../Data/"+filename+".txt")
        W = self.computeSimilarityMatrix(dataset)
        D = self.computeDegreeMatrix(W)
        L = self.computeLaplaciaMatrix(D, W)
        eVal, eVector = self.findEigens(L)
        embeddedSpace = self.sort(eVal, eVector)
        data = self.simulateDataset(embeddedSpace)
        centroids = np.array(self.initializeCentroids(data))
        clusters = self.assignClusters(data, centroids)
        result = hp.sort_result(self, data)
        vs.pca(self, dataset, result)

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
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
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