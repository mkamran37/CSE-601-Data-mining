#Code for generation of association rules

from Apriori import read_data, frequentSetGeneration
import itertools

class AssocRule:

    def __init__(self, min_sup, min_conf, filePath):
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.freqItemsets = frequentSetGeneration(read_data(filePath), min_sup = self.min_sup)
        self.rules = []
        self.associationRules()        

    def associationRules(self):
        for freqItemset in self.freqItemsets:
            if len(freqItemset) > 1:
                self.findRules(freqItemset)

    def findRules(self, freqItemset):
        size = len(freqItemset)
        prev = []
        for item in freqItemset:
            tempItems = set(freqItemset)
            tempItems.remove(item)
            temp = [frozenset(tempItems), frozenset({item})]
            if self.checkMinConfidence(temp):
                self.rules.append(temp)
                prev.append(temp)
        
        while size > 2:
            temp = []
            combinations = itertools.combinations(prev, 2)
            for combination in combinations:
                intersection = combination[0][0] & combination[1][0]
                union = combination[0][1] | combination[1][1]
                if len(intersection) == size - 2:
                    tempList = [frozenset(set(intersection)), frozenset(set(union))]
                    if self.checkMinConfidence(tempList):
                        temp.append(tempList)

            prev = temp
            self.rules += temp
            size -= 1

    def checkMinConfidence(self, rule):
        union = rule[0] | rule[1]
        supCount = int(self.freqItemsets[union])
        if (supCount / int(self.freqItemsets[rule[0]])) >= self.min_conf:
            return True   
        return False

    def template1(self, cond, count, items):
        result = []
        for rule in self.rules:
            if self.checkTemplate1(rule, cond, count, items):
                result.append(rule)
        return result, len(result)

    def checkTemplate1(self, rule, cond, count, items):
        if cond == "RULE" and count == "ANY":
            if len((rule[0] | rule[1]) & set(items)) > 0:
                return True
        elif cond == "RULE" and count == "NONE":
            if len((rule[0] | rule[1]) & set(items)) == 0:
                return True
        elif cond == "RULE" and isinstance(count, int):
            if len((rule[0] | rule[1]) & set(items)) == count:
                return True
        elif cond == "HEAD" and count == "ANY":
            if len(rule[0] & set(items)) > 0:
                return True
        elif cond == "HEAD" and count == "NONE":
            if len(rule[0] & set(items)) == 0:
                return True
        elif cond == "HEAD" and isinstance(count, int):
            if len(rule[0] & set(items)) == count:
                return True
        elif cond == "BODY" and count == "ANY":
            if len(rule[1] & set(items)) > 0:
                return True
        elif cond == "BODY" and count == "NONE":
            if len(rule[1] & set(items)) == 0:
                return True
        elif cond == "BODY" and isinstance(count, int):
            if len(rule[1] & set(items)) == count:
                return True
        return False
        

    def template2(self, cond, count):
        result = []
        for rule in self.rules:
            if self.checkTemplate2(rule, cond, count):
                result.append(rule)
        return result, len(result)

    def checkTemplate2(self, rule, cond, count):
        if cond == "RULE" and (len(rule[0]) + len(rule[1])) >= count:
            return True
        elif cond == "HEAD" and len(rule[0]) >= count:
            return True
        elif cond == "BODY" and len(rule[1]) >= count:
            return True 
        return False

    def template3(self, *args):
        result = []
        for rule in self.rules:
            if self.checkTemplate3(rule, args):
                result.append(rule)
        return result, len(result)

    def checkTemplate3(self, rule, args):
        condition = args[0]
        if condition[0] == '1':
            res1 = self.checkTemplate1(rule, args[1], args[2], args[3])
        elif condition[0] == '2':
            res1 = self.checkTemplate2(rule, args[1], args[2])
        if condition[-1] == '1':
            res2 = self.checkTemplate1(rule, args[4], args[5], args[6])
        elif condition[-1] == '2':
            res2 = self.checkTemplate2(rule, args[3], args[4])

        if condition[1:-1] == 'and':
            return res1 and res2
        elif condition[1:-1] == 'or':
            return res1 or res2

# Saving query results to a file 
def saveResultToFile(result, cnt):
    f = open("results.txt", "w+")
    f.write("Number of rules generated: " + str(cnt) + "\n\n")
    for rule in result:
        f.write(str(set(rule[0])) + " -> " + str(set(rule[1])) + "\n")
    f.close()

if __name__ == "__main__":
    # filePath = "CSE-601/project1/Data/associationruletestdata.txt"
    filePath = "../../Data/assrules.txt"
    min_sup = input("Enter minimum support (in %): ")
    min_conf = input("Enter minimum confidence (in %): ")
    assoc_rule = AssocRule(int(min_sup)/100, int(min_conf)/100, filePath)
    # result, cnt = assoc_rule.associationRules()
    # result, cnt = assoc_rule.template1("HEAD", 2, ['G58_Up', 'G71_Up'])
    # result, cnt = assoc_rule.template2("RULE", 1)
    result, cnt = assoc_rule.template3("1or1", "HEAD", "ANY", ['G81_Down'], "BODY", 1, ['G59_Up'])
    saveResultToFile(result, cnt)
