

class bayes:
    def findProbability(self, data):
        class_map = dict()
        for pt in data:
            if pt.label not in class_map:
                class_map[pt.label] = 1
            else:
                class_map[pt.label] += 1
        res
    def classify(self):
