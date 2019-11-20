import numpy as np
from collections import defaultdict
from helpers import helpers as hp

class knn:
    
    def classify(self, trainData, predictData):
        '''
            input:  trainData - a list of lists of point objects with known labels
                    predictData - a list of point objects with unknown labels
            output: list of point objects with predicted labels
        '''
        for pt in predictData:
            pt.label = self.findLabel(trainData, pt)

    def findLabel(self, trainData, pt, k = 5):
        '''
            input:  trainData - a list of point objects with known labels
                    pt - a single point object whose label has to be determined
            output: label - the predicted label for given point
        '''
        closestNeighbours = sorted(trainData, key = lambda x: np.linalg.norm(x.point - pt.point))[:k]
        majority = defaultdict(int)
        for neighbor in closestNeighbours:
            d = np.linalg.norm(neighbor.point - pt.point)
            if d != 0.0:
                majority[neighbor.groundTruth] += 1/(d**2)
                # majority[neighbor.groundTruth] += 1
            else:
                majority[neighbor.groundTruth] += 1
        if majority[0] > majority[1]:
            return 0
        else:
            return 1