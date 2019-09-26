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
        supCount = float(self.freqItemsets[union])
        if (supCount / float(self.freqItemsets[rule[0]])) >= self.min_conf:
            return True   
        return False

    def template1(self, cond, count, items):
        result = []
        for rule in self.rules:
            if self.checkTemplate1(rule, cond, count, items):
                result.append(rule)
        result = self.removeDuplicates(result)
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
        result = self.removeDuplicates(result)
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
        result = self.removeDuplicates(result)
        return result, len(result)

    def checkTemplate3(self, rule, args):
        condition = args[0]
        if condition[0] == '1' and condition[-1] == '1':
            res1 = self.checkTemplate1(rule, args[1], args[2], args[3])
            res2 = self.checkTemplate1(rule, args[4], args[5], args[6])
        elif condition[0] == '1' and condition[-1] == '2':
            res1 = self.checkTemplate1(rule, args[1], args[2], args[3])
            res2 = self.checkTemplate2(rule, args[4], args[5])
        elif condition[0] == '2' and condition[-1] == '1':
            res1 = self.checkTemplate2(rule, args[1], args[2])
            res2 = self.checkTemplate1(rule, args[3], args[4], args[5])
        elif condition[0] == '2' and condition[-1] == '2':
            res1 = self.checkTemplate2(rule, args[1], args[2])
            res2 = self.checkTemplate2(rule, args[3], args[4])

        if condition[1:-1] == 'and':
            return res1 and res2
        elif condition[1:-1] == 'or':
            return res1 or res2

    def removeDuplicates(self, result):
        resultList = set(tuple(x) for x in result)
        return [list(x) for x in resultList]

# Saving query results to a file 
def saveResultToFile(result, cnt, templateNum, template):
    f = open("results.txt", "a+")
    f.write(templateNum + ": " + template + "\n")
    f.write("Number of rules generated: " + str(cnt) + "\n\n")
    for rule in result:
        f.write(str(set(rule[0])) + " -> " + str(set(rule[1])) + "\n")
    f.write("\n")
    f.close()

if __name__ == "__main__":
    filename = input("enter file name (without extension): ")
    # filePath = "CSE-601/project1/Data/"+filename+".txt"
    filePath = "../../Data/"+filename+".txt"
    min_sup = input("Enter minimum support (in %): ")
    min_conf = input("Enter minimum confidence (in %): ")
    asso_rule = AssocRule(float(min_sup), float(min_conf)/100, filePath)
    (result11, cnt) = asso_rule.template1("RULE", "ANY", ['G59_Up']) 
    saveResultToFile(result11, cnt, "Template1", "RULE|ANY|['G59_Up']")
    (result12, cnt) = asso_rule.template1("RULE", "NONE", ['G59_Up'])
    saveResultToFile(result12, cnt, "Template1", "RULE|NONE|['G59_Up']")
    (result13, cnt) = asso_rule.template1("RULE", 1, ['G59_Up', 'G10_Down']) 
    saveResultToFile(result13, cnt, "Template1", "RULE|1|['G59_Up', 'G10_Down]")
    (result14, cnt) = asso_rule.template1("HEAD", "ANY", ['G59_Up'])
    saveResultToFile(result14, cnt, "Template1", "HEAD|ANY|['G59_Up']")
    (result15, cnt) = asso_rule.template1("HEAD", "NONE", ['G59_Up']) 
    saveResultToFile(result15, cnt, "Template1", "HEAD|NONE|['G59_Up']")
    (result16, cnt) = asso_rule.template1("HEAD", 1, ['G59_Up', 'G10_Down']) 
    saveResultToFile(result16, cnt, "Template1", "HEAD|1|['G59_Up', 'G10_Down']")
    (result17, cnt) = asso_rule.template1("BODY", "ANY", ['G59_Up']) 
    saveResultToFile(result17, cnt, "Template1", "BODY|ANY|['G59_Up']")
    (result18, cnt) = asso_rule.template1("BODY", "NONE", ['G59_Up']) 
    saveResultToFile(result18, cnt, "Template1", "BODY|NONE|['G59_Up']")
    (result19, cnt) = asso_rule.template1("BODY", 1, ['G59_Up', 'G10_Down']) 
    saveResultToFile(result19, cnt, "Template1", "BODY|1|['G59_Up', 'G10_Down']")

    (result21, cnt) = asso_rule.template2("RULE", 3) 
    saveResultToFile(result21, cnt, "Template2", "RULE|3")
    (result22, cnt) = asso_rule.template2("HEAD", 2) 
    saveResultToFile(result22, cnt, "Template2", "HEAD|2")
    (result23, cnt) = asso_rule.template2("BODY", 1)
    saveResultToFile(result23, cnt, "Template2", "BODY|1") 

    (result31, cnt) = asso_rule.template3("1or1", "HEAD", "ANY", ['G10_Down'], "BODY", 1, ['G59_Up']) 
    saveResultToFile(result31, cnt, "Template3", "1or1|HEAD|ANY|['G10_Down']|BODY|1|['G59_UP']")
    (result32, cnt) = asso_rule.template3("1and1", "HEAD", "ANY", ['G10_Down'], "BODY", 1, ['G59_Up']) 
    saveResultToFile(result32, cnt, "Template3", "1and1|HEAD|ANY|['G10_Down']|BODY|1|['G59_Up']")
    (result33, cnt) = asso_rule.template3("1or2", "HEAD", "ANY", ['G10_Down'], "BODY", 2) 
    saveResultToFile(result33, cnt, "Template3", "1or2|HEAD|ANY|['G10_Down']|BODY|2")
    (result34, cnt) = asso_rule.template3("1and2", "HEAD", "ANY", ['G10_Down'], "BODY", 2) 
    saveResultToFile(result34, cnt, "Template3", "1and2|HEAD|ANY|['G10_Down']|BODY|2")
    (result35, cnt) = asso_rule.template3("2or2", "HEAD", 1, "BODY", 2) 
    saveResultToFile(result35, cnt, "Template3", "2or2|HEAD|1|BODY|2")
    (result36, cnt) = asso_rule.template3("2and2", "HEAD", 1, "BODY", 2) 
    saveResultToFile(result36, cnt, "Template3", "2and2|HEAD|1|BODY|2")
