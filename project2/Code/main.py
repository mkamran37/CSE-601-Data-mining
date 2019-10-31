from Spectral import Spectral
from K_means import k_means
from Density import DensityBasedClustering
from helpers import helpers
from External_Index import externalIndex
from visualization import visualization as vs
from Hierarchical import hierarchical 
from Gmm import gmm
import numpy as np

class main:
    
    def kmeans(self, dataset):
        hp = helpers()
        km = k_means()
        datadict = km.convertToDict(dataset)
        k = int(input("Enter number of required clusters: "))
        centroids = np.array(km.initializeCentroids(datadict, k))
        iterations = int(input("Enter number of max iterations: "))
        centroids = km.assignClusters(dataset, centroids, iterations)
        result = hp.sort_result(dataset)
        return dataset, result, centroids

    def hrClustering(self):
        fileName = input("Enter data file name (without extension): ")
        # filePath = "CSE-601/project2/Data/"+ fileName + ".txt"
        filePath = "../Data/"+fileName+".txt"
        numClusters = int(input("Enter number of required clusters: "))
        hr = hierarchical(filePath, numClusters)
        dataset, predicted, ids = hr.agglomerative()
        return dataset, predicted, ids, fileName

    def gmmClustering(self):
        flag = int(input("Enter 1 for Kmeans initialization else Enter 0: "))
        centroids = []
        if flag == 1:
            hp = helpers()
            dataset, fileName = hp.get_file()
            _, _, centroids = m.kmeans(dataset)
        else:
            fileName = input("Enter data file name (without extension): ")
        filePath = "../Data/"+fileName+".txt"
        # filePath = "CSE-601/project2/Data/"+ fileName + ".txt"
        g = gmm(filePath, centroids)
        dataset, predicted, ids = g.emAlgorithm()
        return dataset, predicted, ids, fileName

    def spectral(self, dataset):
        sp = Spectral()
        sigma = float(input("Enter the value for sigma: "))
        W = sp.computeSimilarityMatrix(dataset, sigma)
        D = sp.computeDegreeMatrix(W)
        L = sp.computeLaplacianMatrix(D, W)
        eVal, eVector = sp.findEigens(L)
        k = int(input("Enter the number of required clusters: "))
        embeddedSpace = sp.sort(eVal, eVector)
        data = sp.simulateDataset(embeddedSpace)
        centroids = np.array(sp.initializeCentroids(data, k))
        max_iterations = int(input("Enter maximum number of iterations: "))
        clusters = sp.assignClusters(data, centroids, max_iterations)
        dd = sp.convertData(data, dataset)
        result = hp.sort_result(dd)
        return dd, result
    
    def density(self, dataset):
        hp = helpers()
        db = DensityBasedClustering()
        distance = db.findDistanceMatrix(dataset)
        eps = float(input("Enter the value for epsilon parameter: "))
        minpts = int(input("Enter the minimum number of pts for a core point: "))
        db.dbScan(dataset, eps=eps, minpts=minpts, distance=distance)
        result = hp.sort_result(dataset)
        return dataset, result

if __name__ == "__main__":
    choice = int(input("\nPress 1 for k-means\nPress 2 for Hierarchical Clustering\nPress 3 for Density based Clustering\nPress 4 for Gaussian Mixture Model Clustering\nPress 5 for Spectral Clustering\n"))
    m = main()
    hp = helpers()
    if choice == 1:
        dataset, filename = hp.get_file()
        dataset, result, centroids = m.kmeans(dataset)
        dataset, ids, predicted = hp.create_pd(dataset)
    elif choice == 2:
        dataset, predicted, ids, filename = m.hrClustering()
    elif choice == 3:
        dataset, filename = hp.get_file()
        dataset, result = m.density(dataset)
        dataset, ids, predicted = hp.create_pd(dataset)
    elif choice == 4:
        dataset, predicted, ids, filename = m.gmmClustering()
    else:
        dataset, filename = hp.get_file()
        dataset, result = m.spectral(dataset)
        dataset, ids, predicted = hp.create_pd(dataset)
    hp.calculateCoeff(predicted, filename, ids)
    if dataset[0].shape[0] == 2:
        vs().visualize(dataset, predicted, ids)
    else:
        vs().pca(dataset, predicted, ids)
