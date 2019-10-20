import numpy as np
from scipy.spatial import distance
import random
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class gene_representation:
    def __init__(self, geneId = -1, clusterNumber = -1, gene = None):
        self.geneID = geneId
        self.clusterNumber = clusterNumber
        if gene is None:
            self.gene = list()
        else:
            self.gene = gene
class k_means:
    def __init__(self):
        filename = input("enter file name (without extension)")
        dataset = self.read_data("../Data/"+filename+".txt")
        centroids = np.array(self.initializeCentroids(dataset))
        self.assignClusters(dataset, centroids)
        result = self.sort_result(dataset)
        # self.pca(dataset, result)
        truth = self.groundtruth("../Data/"+filename+".txt")
        # self.findJaccard(predicted, truth)
    
    def sort_result(self, datasets):
        cluster = defaultdict(list)
        dictlist = []
        for dataset in datasets:
            cluster[int(dataset.geneID)].append(int(dataset.clusterNumber))
        for key, value in cluster.items():
            temp = [key,value[0]]
            dictlist.append(temp)
        dictlist.sort(key= lambda x:x[0])
        return dictlist
    
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

    def read_data(self, filepath):
        #Read data from text file as numpy ndarray
        data = np.genfromtxt(filepath, dtype='double', delimiter="\t")
        dataset = list()
        for i in range(data.shape[0]):
            tmp = list()
            gene = gene_representation()
            for j in range(data.shape[1]):
                if j == 0:
                    gene.geneID = int(data[i][0])
                elif j == 1:
                    continue
                else:
                    tmp.append(data[i][j])
            gene.gene = tmp
            dataset.append(gene)
        return dataset
    
    def pca(self, datasett, result):
        pca = PCA(n_components=2, svd_solver='full')
        dataset = [data.gene for data in datasett]
        pca.fit(dataset)
        r = np.array(result)
        pca_matrix = pca.transform(dataset)
        df = pd.DataFrame(data = np.concatenate((pca_matrix, r[:,1:2]), axis = 1), columns=['PC1','PC2','Cluster'])
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        plt.show()
    
    def scatter_plot(self, df):
        pt = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        # pt.fig.suptitle("Algorithm: " + algorithm + "  " + "Dataset: " + dataFIle)
        plt.show()

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
            dist = distance.euclidean(gene.gene, centroid)
            if dist < min_dist:
                min_dist = dist
                cluster = i+1
        gene.clusterNumber = int(cluster)
        clusters[cluster].append(gene.gene)
        return clusters

    def findClusterCentroid(self, centroids, clusters):
        for i,key in enumerate(clusters):
            centroids[i] = np.array(clusters[key], dtype=np.float64).mean(axis=0)
        return centroids

if __name__ == "__main__":
    k = k_means()


            
