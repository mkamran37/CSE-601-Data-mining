#Code for generation of association rules
from Apriori import read_data, frequentSetGeneration
import itertools
class AssocRule:

    freqItemsets = {frozenset({'G9_Down', 'G58_Up'}) : 12, frozenset({'G87_Down', 'G86_Up'}) : 15}
    def __init__(self, min_sup, min_conf, filePath):
        self.min_sup = min_sup
        self.min_conf = min_conf
        # self.freqItemsets = frequentSetGeneration(read_data(filePath))

    def associationRules(self):
        for freqItemset in freqItemsets:
            findRules(freqItemset)

    def findRules(self, freqItemset):
        size = len(freqItemset)
        rules = []
        for item in freqItemset:
            tempItems = freqItemset
            tempItems.remove(item)
            temp = list(tempItems, frozenset(item))
            if checkMinConfidence(temp):
                rules.add(temp)
        
        prev = rules
        while size > 2:
            temp = []
            combinations = itertools.combinations(prev, 2)
            for combination in combinations:
                intersection = combination[0][0] & combination[1][0]
                union = combination[0][1] | combination[1][1]
                tempList = list(frozenset(intersection), frozenset(union))
                if checkMinConfidence(tempList):
                    temp.add(tempList)

            prev = temp
            rules += temp
            size -= 1
        
        return rules 

            
    def checkMinConfidence(self, rule):
        conf = self.freqItemsets.get(rule[0] | rule[1])
        if (conf / self.freqItemsets.get(rule[0])) <= self.min_conf:
            return True   
        return False

if __name__ == "__main__":
    filePath = "CSE-601/project1/Data/associationruletestdata.txt"
    assoc_rule = AssocRule(0.5, 0.7, filePath)