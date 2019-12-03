import numpy as np
from collections import defaultdict

class knn:
    
    def classify(self, trainData, predictData, k):
        '''
            type:  trainData - a list of lists of point objects with known labels
                    predictData - a list of point objects with unknown labels
            rtype: list of point objects with predicted labels
        '''
        for pt in predictData:
            pt.label = self.findLabel(trainData, pt, k)

    def findLabel(self, trainData, pt, k):
        '''
            type:  trainData - a list of point objects with known labels
                    pt - a single point object whose label has to be determined
            rtype: label - the predicted label for given point
        '''
        closestNeighbours = sorted(trainData, key = lambda x: np.linalg.norm(x.point - pt.point))[:k]
        majority = defaultdict(int)
        for neighbor in closestNeighbours:
            majority[neighbor.groundTruth] += 1
        if majority[0] > majority[1]:
            return 0
        else:
            return 1