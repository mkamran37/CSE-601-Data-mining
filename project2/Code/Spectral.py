import numpy as np
from collections import defaultdict
from math import e
from point import Point
from visualization import visualization as vs
from helpers import helpers as hp
from K_means import k_means as km
from External_Index import externalIndex

class Spectral:
    def __init__(self):
        filename = input("enter file name (without extension)")
        dataset = hp.read_data(self, "../Data/"+filename+".txt")
        sigma = int(input("Enter the value for sigma: "))
        W = self.computeSimilarityMatrix(dataset, sigma)
        D = self.computeDegreeMatrix(W)
        L = self.computeLaplaciaMatrix(D, W)
        eVal, eVector = self.findEigens(L)
        k = int(input("Enter the number of required clusters: "))
        embeddedSpace = self.sort(eVal, eVector, k)
        data = self.simulateDataset(embeddedSpace)
        centroids = np.array(self.initializeCentroids(data, k))
        max_iterations = int(input("Enter maximum number of iterations: "))
        clusters = self.assignClusters(data, centroids, max_iterations)
        dd = self.convertData(data, dataset)
        result = hp.sort_result(self, dd)
        # vs.pca(self, dd, result)
        ids, predicted = hp.create_pd(self, dd)
        groundTruth = np.genfromtxt("../Data/"+filename+".txt", delimiter="\t", dtype=str, usecols=1)
        coeff = externalIndex(predicted, groundTruth, ids)
        rand, jaccard = coeff.getExternalIndex()
        print("RAND COEFFICIENT: {}".format(rand))
        print("JACCARD COEFFICIENT: {}".format(jaccard))

    def simulateDataset(self, dataset):
        '''
            input: dataset - top k eigen vectors
            output: data - a dictionary containing key as gene ID and value as the corresponding Point object
        '''
        data = dict()
        for i in range(len(dataset)):
            pt = Point()
            pt.id = i+1
            pt.point = dataset[i]
            data[i+1] = pt
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
                dist = np.linalg.norm(point.point - p.point)
                similarityMatrix[point.id-1][p.id-1] = np.exp(-dist**2/(2.*(sigma**2.)))
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

    def initializeCentroids(self, dataset, k=5):
        '''
        input:  dataset - a dictionary containing key as gene ID and value as the corresponding Point object
                k - number of desired clusters
        output: centroids - k number of initial random centroids
        '''
        centroids = list()
        while k > 0:
            idx = int(input("enter id: "))
            centroids.append(dataset[idx].point)
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
                self.find_cluster(centroids, dataset[i+1], clusters)
            centroids = self.findClusterCentroid(centroids, clusters)
            j+=1
        return clusters
        
    def find_cluster(self, centroids, gene, clusters):
        min_dist = float('inf')
        cluster = 0
        for i,centroid in enumerate(centroids):
            dist = np.linalg.norm(gene.point - centroid)
            if dist <= min_dist:
                min_dist = dist
                cluster = i+1
        gene.cluster = int(cluster)
        clusters[cluster].append(gene)
        return clusters

    def findClusterCentroid(self, centroids, clusters):
        for i,key in enumerate(clusters):
            tmp = list()
            tmp = [point.point for point in clusters[key]]
            centroids[i] = np.array(tmp, dtype=np.float64).mean(axis=0)
        return centroids
    
    def convertData(self, data, dataset):
        '''
        input: data- (eigenvector) a dictionary of gene ID vs. list of Point objects
            dataset- (original points) the original data containing point objects
        output: dd- a list containing only of Point objects
        '''
        dd = list()
        for point in dataset:
            tmp = data[point.id]
            point.cluster = tmp.cluster
            dd.append(point)
        return dd