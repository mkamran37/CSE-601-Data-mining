from collections import defaultdict
from helpers import helpers as hp
from math import e

class bayes:
    def findClassPriorProbability(self, data):
        '''
            :type  data- a list of Point objects
            :rtype res- a dictionary with key as class label and value as its probability
        '''
        class_map = defaultdict(int)
        for pt in data:
            class_map[pt.groundTruth] += 1
        res = dict()
        for key in class_map:
            res[key] = class_map[key]/len(data)
        return res

    def segregateClasses(self, data):
        classes = defaultdict(list)
        for point in data:
            classes[point.groundTruth].append(point)
        return classes

    def countOccurence(self, pt, index, data):
        count = 0
        for tmp in data:
            if tmp.categoricalData[index] == pt:
                count+=1
        return count

    def findDescriptorPosteriorProbabilites(self, classes, td):
        occurences = defaultdict(int)
        mean, stdDeviation = defaultdict(dict), defaultdict(dict)
        for key in classes:
            tmp = classes[key]
            mean[key], stdDeviation[key] = hp().standardizeBayes(tmp)
            for pt in tmp:
                for index, i in enumerate(pt.categoricalData):
                    if (i, key) not in occurences:
                        count = self.countOccurence(i, index, tmp)
                        occurences[(i, key)] = count/len(tmp)
        return occurences, mean, stdDeviation

    def classify(self, predictData, classPriorProbabilities, occurences, mean, stdDeviation):
        for pt in predictData:
            pt.label = self.bayesProbabilty(pt, classPriorProbabilities, occurences, mean, stdDeviation)
    
    def classify_demo(self, predictData, classPriorProbabilities, occurences, mean, stdDeviation):
        for pt in predictData:
            return self.bayesProbabiltyDemo(pt, classPriorProbabilities, occurences, mean, stdDeviation)

    def bayesProbabilty(self, point, ph, occurences, mean, stdDeviation):
        maxProbability = float('-inf')
        label = -1
        for key in ph:
            phi = ph[key]
            probability = 1.0
            for index, i in enumerate(point.point):
                den = 2*(22/7)*(stdDeviation[key][index]**2)
                num = ((i-mean[key][index])**2)/(2*(stdDeviation[key][index]**2))
                probability*=(1/den**0.5)*(e**(-1*num))
            for index, i in enumerate(point.categoricalData):
                probability*=occurences[(i, key)]
            probability*=phi
            if probability >= maxProbability:
                maxProbability = probability
                label = key
        return label
    
    def bayesProbabiltyDemo(self, point, ph, occurences, mean, stdDeviation):
        probabilities = defaultdict()
        for key in ph:
            phi = ph[key]
            probability = 1.0
            for i in point.categoricalData:
                probability*=occurences[(i, key)]
            probability*=phi
            probabilities[key] = probability
        return probabilities


