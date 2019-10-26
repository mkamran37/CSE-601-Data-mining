import numpy as np
import pandas as pd
from scipy import spatial
from External_Index import externalIndex

class hierarchical:

    def __init__(self, filePath, numClusters):
        self.numClusters = numClusters
        dataMatrix, self.geneIds, self.groundTruth = self.readData(filePath)
        self.gt_incidenceMatrix = self.getGroundTruthIncidenceMatrix()
        self.distanceMatrix = self.getDistanceMatrix(dataMatrix)

    def readData(self, filePath):
        #Read data from text file as numpy ndarray
        data = np.genfromtxt(filePath, dtype='float', delimiter="\t")
        data = np.delete(data, [0,1], axis=1)
        geneIds = np.genfromtxt(filePath, delimiter="\t", dtype=str, usecols=0)
        groundTruth = np.genfromtxt(filePath, delimiter="\t", dtype=str, usecols=1)
        df = pd.DataFrame(data)
        return df, geneIds, groundTruth

    def getGroundTruthIncidenceMatrix(self):
        incidenceMatrix = np.zeros(shape=(len(self.groundTruth), len(self.groundTruth)))
        for i in range(0, len(self.groundTruth)):
            for j in range(0, len(self.groundTruth)):
                if self.groundTruth[i] == self.groundTruth[j]:
                    incidenceMatrix[i][j] = 1

        incidenceMatrix = pd.DataFrame(incidenceMatrix, index=self.geneIds, columns=self.geneIds)
        return incidenceMatrix

    def getDistanceMatrix(self, dataMatrix):
        distanceMatrix = spatial.distance.cdist(dataMatrix, dataMatrix, metric='euclidean')
        distanceMatrix = pd.DataFrame(distanceMatrix, index=self.geneIds, columns=self.geneIds)
        return distanceMatrix

    def agglomerative(self):
        clusterMatrix = self.distanceMatrix.replace(0, np.inf)
        while(clusterMatrix.shape[0] > self.numClusters):
            print(clusterMatrix.shape)
            # getting index of minimum value in distance matrix
            # print(clusterMatrix)
            minclusterIndex1, minClusterIndex2 = clusterMatrix.stack().idxmin()
            # print(clusterMatrix.loc[[minclusterIndex1], [minClusterIndex2]])
            # removing the rows and columns of the min value
            clusterMatrix.drop([minclusterIndex1, minClusterIndex2], axis = 1, inplace = True)
            clusterMatrix.drop([minclusterIndex1, minClusterIndex2], inplace = True)

            # new index
            newClusterIndex = minclusterIndex1 + "-" + minClusterIndex2
            # creating new row 
            rowDf = pd.DataFrame(np.full((1, clusterMatrix.shape[1]), np.inf), columns=clusterMatrix.columns)
            rowDf.rename(index={0:newClusterIndex}, inplace=True)

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

            clusterMatrix = pd.concat([clusterMatrix, rowDf], sort=False)
            clusterMatrix = pd.concat([clusterMatrix, colDf], axis=1, sort=False)
            clusterMatrix[newClusterIndex][newClusterIndex] = np.inf
            
        self.clusters = list(clusterMatrix.index.values)
        # print(self.clusters)
        self.predictedMatrix = self.getPredictedMatrix()

    def getPredictedMatrix(self):
        predictedMatrix = np.zeros(shape=(len(self.distanceMatrix), 1))
        predictedMatrix = pd.DataFrame(predictedMatrix, index=self.geneIds, columns=['clusterNum'])

        for i in range(0, self.numClusters):
            for data in self.clusters[i].split('-'):
                predictedMatrix.at[data, 'clusterNum'] = i+1

        return predictedMatrix

if __name__ == "__main__":
    fileName = input("Enter data file name (without extension): ")
    filePath = "CSE-601/project2/Data/"+ fileName + ".txt"
    numClusters = int(input("Enter the number of clusters: "))
    hr = hierarchical(filePath, numClusters)
    hr.agglomerative()
    extIndex = externalIndex(hr.predictedMatrix, hr.groundTruth, hr.geneIds)
    rand, jaccard = extIndex.getExternalIndex()
    print(rand, jaccard)
