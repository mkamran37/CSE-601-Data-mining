import numpy as np
from scipy.spatial import distance
import random
from collections import defaultdict
from point import Point
from visualization import visualization as vs
from helpers import helpers as hp

class k_means:

    def convertToDict(self, dataset):
        dataDict = dict()
        for i in range(len(dataset)):
            dataDict[i+1] = dataset[i]
        return dataDict
   
    def assignClusters(self, dataset, centroids, iterations = 10):
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
        return centroids

    def initializeCentroids(self, dataset, k):
        centroids = list()
        cluster = 0
        print("Enter {} number of initial centroids: ".format(k))
        while cluster < k:
            point = dataset[int(input("Enter id: "))]
            centroids.append(point.point)
            cluster+=1
        return centroids
    
    def find_cluster(self, centroids, data, clusters):
        min_dist = float('inf')
        cluster = 0
        for i,centroid in enumerate(centroids):
            dist = distance.euclidean(data.point, centroid)
            if dist < min_dist:
                min_dist = dist
                cluster = i+1
        data.cluster = int(cluster)
        clusters[cluster].append(data.point)
        return clusters

    def findClusterCentroid(self, centroids, clusters):
        for i,key in enumerate(clusters):
            centroids[i] = np.array(clusters[key], dtype=np.float64).mean(axis=0)
        return centroids