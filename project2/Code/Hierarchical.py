import numpy as np
import pandas as pd
from scipy import spatial

class hierarchical:

    def __init__(self, filePath, numClusters):
        self.numClusters = numClusters
        self.dataMatrix, self.dataIds = self.readData(filePath)
        self.distanceMatrix = self.getDistanceMatrix()

    def readData(self, filePath):
        #Read data from text file as numpy ndarray
        data = np.genfromtxt(filePath, dtype='float', delimiter="\t")
        data = np.delete(data, [0,1], axis=1)
        dataIds = np.genfromtxt(filePath, delimiter="\t", dtype=str, usecols=0)
        dataDf = pd.DataFrame(data)
        return dataDf, dataIds

    def getDistanceMatrix(self):
        # To calculate the initial distance matrix
        distanceMatrix = spatial.distance.cdist(self.dataMatrix, self.dataMatrix, metric='euclidean')
        distanceMatrix = pd.DataFrame(distanceMatrix, index=self.dataIds, columns=self.dataIds)
        return distanceMatrix

    def agglomerative(self):
        print("\nRunning hierarchical clustering (will take some time) ....................")
        clusterMatrix = self.distanceMatrix.replace(0, np.inf)
        while(clusterMatrix.shape[0] > self.numClusters):
            # finding the row and column with min distance value
            minclusterIndex1, minClusterIndex2 = clusterMatrix.stack().idxmin()
            # Deleting the row and column
            clusterMatrix.drop([minclusterIndex1, minClusterIndex2], axis = 1, inplace = True)
            clusterMatrix.drop([minclusterIndex1, minClusterIndex2], inplace = True)

            # new index
            newClusterIndex = minclusterIndex1 + "-" + minClusterIndex2
            # creating new row 
            rowDf = pd.DataFrame(np.full((1, clusterMatrix.shape[1]), np.inf), columns=clusterMatrix.columns)
            rowDf.rename(index={0:newClusterIndex}, inplace=True)
            # creating new column
            colDf = pd.DataFrame(np.full((clusterMatrix.shape[0], 1), np.inf), index=clusterMatrix.index)
            colDf.rename(columns={0:newClusterIndex}, inplace=True)

            for cluster in list(rowDf):
                minValue = float('Inf')
                for data in cluster.split('-'):
                    for j in minclusterIndex1.split('-'):
                        minValue = min(minValue, self.distanceMatrix.at[data, j])

                    for j in minClusterIndex2.split('-'):
                        minValue = min(minValue, self.distanceMatrix.at[data, j])

                rowDf[cluster][newClusterIndex] = minValue
                colDf[newClusterIndex][cluster] = minValue

            clusterMatrix = pd.concat([clusterMatrix, rowDf])
            clusterMatrix = pd.concat([clusterMatrix, colDf], axis=1)
            clusterMatrix[newClusterIndex][newClusterIndex] = np.inf
            
        self.clusters = list(clusterMatrix.index.values)
        # print(self.clusters)
        return self.dataMatrix, self.getPredictedMatrix(), self.dataIds

    def getPredictedMatrix(self):
        predictedMatrix = np.zeros(shape=(len(self.distanceMatrix), 1))
        predictedMatrix = pd.DataFrame(predictedMatrix, index=self.dataIds, columns=['Cluster'])

        for i in range(0, self.numClusters):
            for data in self.clusters[i].split('-'):
                predictedMatrix.at[data, 'Cluster'] = i+1

        return predictedMatrix
