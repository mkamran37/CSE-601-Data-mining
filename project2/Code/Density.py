import numpy as np
from scipy.spatial import distance
from collections import defaultdict
from point import Point
from visualization import visualization as vs
from helpers import helpers as hp
from External_Index import externalIndex


class DensityBasedClustering:
       
    def findDistanceMatrix(self, dataset):
        distanceMatrix = [[0 for x in range(len(dataset))] for y in range(len(dataset))]
        for point in dataset:
            for p in dataset:
                distanceMatrix[point.id-1][p.id-1] = distance.euclidean(point.point, p.point)
        return distanceMatrix
    
    def dbScan(self, dataset, eps=1.0, minpts=5, distance=None, points=None):
        clusterNumber = 0
        clusters = defaultdict(list)
        visited = set()
        clustered = set()
        for point in dataset:
            if point not in visited: 
                visited.add(point)
                neigbors = self.regionQuery(point, eps, distance, dataset)
                if len(neigbors)+1 < minpts:
                    clustered.add(point)
                else:
                    clusterNumber+=1
                    self.expandCluster(point, neigbors, clusters, eps, minpts, clusterNumber, visited, distance, dataset, clustered)
        return clusters
    
    def expandCluster(self, point, neighbors, clusters, eps, minpts, clusterNumber, visited, distance, dataset, clustered):
        clusters[clusterNumber].append(point)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                newNeighbours = self.regionQuery(neighbor, eps, distance, dataset)
                if len(newNeighbours)+1 >= minpts:
                    for n in newNeighbours:
                        neighbors.append(n)
            if neighbor not in clustered:
                clustered.add(neighbor)
                clusters[clusterNumber].append(neighbor)
                neighbor.cluster = clusterNumber
    
    def regionQuery(self, neighbor, eps, distance, points):
        result = list()
        for point in points:
            if distance[neighbor.id-1][point.id-1] < eps:
                result.append(point)
        return result