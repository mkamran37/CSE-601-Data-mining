from collections import defaultdict

class bayes:
    def findClassPriorProbability(self, data):
        '''
            input:  data- a list of Point objects
            output: res- a dictionary with key as class label and value as its probability
        '''
        class_map = defaultdict(int)
        for pt in data:
            class_map[pt.label] += 1
        res = dict()
        for key in class_map:
            res[key] = class_map[key]/len(data)
        return res
    
    def findDescriptorPriorProbabilities(self, data):
        product = 1
        for row in data:
            p = 1
            for i in range(len(row.point)):
                p*=self.findProbability(row.point[i], i, data)
            product*=(p/len(data))
        return product
    
    def findProbability(self, attr, index, data):
        counter = 0
        for i in range(len(data)):
            if data[i].point[index] == attr:
                counter+=1
        return counter/len(data)

    def segregateClasses(self, data):
        classes = defaultdict(list)
        for point in data:
            classes[point.label].append(point)
        return classes

    def countOccurence(self, pt, index, data):
        count = 0
        for tmp in data:
            if tmp.point[index] == pt:
                count+=1
        return count

    def findDescriptorPosteriorProbabilites(self, classes):
        res = defaultdict(int)
        for key in classes:
            tmp = classes[key]
            for pt in tmp:
                for index, i in enumerate(pt.point):
                    if (i, key) not in res:
                        res[(i, key)] = self.countOccurence(i, index, tmp)/len(tmp)
        return res
    
    def classify(self, predictData, classPriorProbabilities, descriptorPosteriorProbabilites):
        for pt in predictData:
            pt.label = self.bayesProbabilty(pt.point, classPriorProbabilities, descriptorPosteriorProbabilites)
    
    def bayesProbabilty(self, points, ph, pxh):
        maxProbability = -1.0
        label = -1
        for key in ph:
            phi = ph[key]
            probability = 1.0
            for pt in points:
                if (pt, key) in pxh:
                    probability*=pxh[(pt, key)]
            if probability != 1.0:
                probability*=phi
            if probability >= maxProbability:
                maxProbability = probability
                label = key
        return label


