import numpy as np
from scipy.spatial import distance
import random
from collections import defaultdict
from point import Point
from visualization import visualization as vs
from helpers import helpers as hp

class k_means:
    
    def __init__(self):
        filename = input("enter file name (without extension)")
        dataset = hp.read_data(self, "../Data/"+filename+".txt")
        centroids = np.array(self.initializeCentroids(dataset))
        self.assignClusters(dataset, centroids)
        result = hp.sort_result(self, dataset)
        vs.pca(self, dataset, result)
        # truth = self.groundtruth("../Data/"+filename+".txt")
        # self.findJaccard(predicted, truth)
    
    def findJaccard(self, predicted, truth):
        for key in predicted:
            l1=truth[key]
            l2=predicted[key]
            intersection = [list(x) for x in set(tuple(x) for x in l1).intersection(set(tuple(x) for x in l2))]
            # union = len(predicted[key]) + len(truth[key]) - len(intersection)
            union = len(predicted[key]) + len(truth[key])
            print(len(intersection)/union)

    def groundtruth(self, filepath):
        data = np.genfromtxt(filepath, dtype='double', delimiter="\t")
        dataset = dict()
        for i in range(data.shape[0]):
            tmp = list()
            for j in range(data.shape[1]):
                if j==0:
                    continue
                elif j==1:
                    if data[i][j] not in dataset and data[i][j] != -1:
                        dataset[int(data[i][j])] = list()
                else:
                    tmp.append(data[i][j])
            if data[i][1] != -1:
                dataset.get(int(data[i][1])).append(tmp)
        return dataset
        # print(dataset)
        # self.pca(dataset)

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

    def initializeCentroids(self, dataset, k = 5):
        centroids = list()
        cluster = 0
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



            
