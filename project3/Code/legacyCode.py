    def findDescriptorPriorProbabilities(self, data):
        product = 1.0
        for row in data:
            p = 1.0
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