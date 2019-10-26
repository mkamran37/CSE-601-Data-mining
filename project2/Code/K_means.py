import numpy as np
from scipy.spatial import distance
import random
from collections import defaultdict
from point import Point
from visualization import visualization as vs
from helpers import helpers as hp
from External_Index import externalIndex

class k_means:
    
    def __init__(self):
        filename = input("enter file name (without extension)")
        dataset = hp.read_data(self, "../Data/"+filename+".txt")
        k = int(input("Enter number of required clusters: "))
        centroids = np.array(self.initializeCentroids(dataset, k))
        iterations = int(input("Enter number of max iterations: "))
        self.assignClusters(dataset, centroids, iterations)
        result = hp.sort_result(self, dataset)
        # vs.pca(self, dataset, result)
        ids, predicted = hp.create_pd(self, dataset)
        groundTruth = np.genfromtxt("../Data/"+filename+".txt", delimiter="\t", dtype=str, usecols=1)
        coeff = externalIndex(predicted, groundTruth, ids)
        rand, jaccard = coeff.getExternalIndex()
        print("RAND COEFFICIENT: {}".format(rand))
        print("JACCARD COEFFICIENT: {}".format(jaccard))

    def assignClusters(self, dataset, centroids, iterations = 200):
        # prevCentroids = np.empty_like(centroids)
        clusters = defaultdict(list)
        j = 0
        # while not np.equals(prevCentroids, centroids)
        while j < iterations:
            # prevCentroids = centroids
            clusters = defaultdict(list)
            for i in range(len(dataset)):
               clusters = self.find_cluster(centroids, dataset[i], clusters)
            centroids = self.findClusterCentroid(centroids, clusters)
            j+=1

    def initializeCentroids(self, dataset, k):
        centroids = list()
        cluster = 0
        print("Enter {} number of initial centroids: ".format(k))
        while cluster < k:
            inputs = list(map(float,input().split()))
            centroids.append(inputs)
            cluster+=1
        return centroids
    
    def find_cluster(self, centroids, gene, clusters):
        min_dist = float('inf')
        cluster = 0
        for i,centroid in enumerate(centroids):
            dist = distance.euclidean(gene.point, centroid)
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