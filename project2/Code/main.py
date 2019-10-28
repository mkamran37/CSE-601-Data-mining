from Spectral import Spectral
from K_means import k_means
from Density import DensityBasedClustering
from helpers import helpers as hp
from External_Index import externalIndex
from visualization import visualization as vs
import numpy as np

class main:
    
    def kmeans(self, dataset):
        km = k_means()
        datadict = km.convertToDict(dataset)
        k = int(input("Enter number of required clusters: "))
        centroids = np.array(km.initializeCentroids(datadict, k))
        iterations = int(input("Enter number of max iterations: "))
        centroids = km.assignClusters(dataset, centroids, iterations)
        result = hp.sort_result(self, dataset)
        # vs.pca(self, dataset, result)
        return dataset, result, centroids

    def spectral(self, dataset):
        sp = Spectral()
        sigma = int(input("Enter the value for sigma: "))
        W = sp.computeSimilarityMatrix(dataset, sigma)
        D = sp.computeDegreeMatrix(W)
        L = sp.computeLaplacianMatrix(D, W)
        eVal, eVector = sp.findEigens(L)
        k = int(input("Enter the number of required clusters: "))
        embeddedSpace = sp.sort(eVal, eVector, k)
        data = sp.simulateDataset(embeddedSpace)
        centroids = np.array(sp.initializeCentroids(data, k))
        max_iterations = int(input("Enter maximum number of iterations: "))
        clusters = sp.assignClusters(data, centroids, max_iterations)
        dd = sp.convertData(data, dataset)
        result = hp.sort_result(self, dd)
        return dd, result
    
    def density(self, dataset):
        db = DensityBasedClustering()
        distance = db.findDistanceMatrix(dataset)
        eps = int(input("Enter the value for epsilon prameter: "))
        minpts = int(input("Enter the minimum number of pts for a core point: "))
        db.dbScan(dataset, eps=eps, minpts=minpts, distance=distance)
        result = hp.sort_result(self, dataset)
        return dataset, result

if __name__ == "__main__":
    choice = int(input("\nPress 1 for k-means\nPress 2 for Density based Clustering\nPress 3 for Spectral clustering\n"))
    m = main()
    dataset, filename = hp.get_file()
    if choice == 1:
        dataset, result, centroids = m.kmeans(dataset)
    elif choice == 2:
        dataset, result = m.density(dataset)
    else:
        dataset, result = m.spectral(dataset)
    hp.calculateCoeff(dataset, filename)
    vs.pca(m, dataset, result)
