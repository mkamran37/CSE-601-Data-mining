import numpy as np
import pandas as pd

class externalIndex:

    def __init__(self, predictedMatrix, groundTruthMatrix, dataIds):
        print("Finding Rand and Jaccard Coefficient ....................")
        self.dataIds = dataIds
        self.groundTruth = self.getGroundTruthIncidenceMatrix(groundTruthMatrix)
        self.predicted = self.getPredictedIncidenceMatrix(predictedMatrix)
        
    def getGroundTruthIncidenceMatrix(self, groundTruthMatrix):
        incidenceMatrix = np.zeros(shape=(len(groundTruthMatrix), len(groundTruthMatrix)))
        for i in range(0, len(groundTruthMatrix)):
            for j in range(0, len(groundTruthMatrix)):
                if groundTruthMatrix[i] == groundTruthMatrix[j]:
                    incidenceMatrix[i][j] = 1

        incidenceMatrix = pd.DataFrame(incidenceMatrix, index=self.dataIds, columns=self.dataIds)
        return incidenceMatrix

    def getPredictedIncidenceMatrix(self, predictedMatrix):
        incidenceMatrix = np.zeros(shape=(len(predictedMatrix), len(predictedMatrix)))
        incidenceMatrix = pd.DataFrame(incidenceMatrix, index=self.dataIds, columns=self.dataIds)

        for i in range(0, len(self.dataIds)):
            for j in range(i, len(self.dataIds)):
                index1 = self.dataIds[i]
                index2 = self.dataIds[j]
                if predictedMatrix.at[index1, 'Cluster'] == predictedMatrix.at[index2, 'Cluster']:
                    incidenceMatrix.at[index1, index2] = 1
                    incidenceMatrix.at[index2, index1] = 1

        return incidenceMatrix

    def getExternalIndex(self):
        m11 = 0
        m00 = 0
        m10 = 0
        m01 = 0

        # print(predicted, groundTruth)
        for index1 in self.dataIds:
            for index2 in self.dataIds:
                if self.predicted.at[index1, index2] == 1 and self.groundTruth.at[index1, index2] == 1:
                    m11 += 1
                elif self.predicted.at[index1, index2] == 0 and self.groundTruth.at[index1, index2] == 0:
                    m00 += 1
                elif self.predicted.at[index1, index2] == 1 and self.groundTruth.at[index1, index2] == 0:
                    m10 += 1
                elif self.predicted.at[index1, index2] == 0 and self.groundTruth.at[index1, index2] == 1:
                    m01 += 1

        rand = (m11 + m00) / (m11 + m00 + m10 + m01)
        jaccard = (m11) / (m11 + m10 + m01)
        return rand, jaccard