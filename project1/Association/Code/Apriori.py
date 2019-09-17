#ALGORITHM:
#Ci is the candidate set, min_sup = minimum support, min_conf = minimum confidence
#Li is the set of candidates whose support >=min_sup
#K = 1
#Step1: Generate all the 1-itemset along with their support
#Step2: Remove all itemsets whose support<min_sup
#Step3: K=K+1
#repeat the above steps till Lk-1 == Lk;
import numpy as np
import itertools as it

def read_data(filepath):
    #Read data from text file as numpy ndarray
    data = np.genfromtxt(filepath, dtype='str', delimiter="\t")
    data = np.delete(data, -1, axis=1)
    for i in range(data.shape[0]):
        t = 0
        for j in range(data.shape[1]):
            data[i][j] = 'G'+str(t)+'_'+data[i][j]
            t+=1
    return np.ndarray.tolist(data)

def frequentSetGeneration(data, min_sup = 0.5):
    K = 1
    L = []
    size = len(data)
    L = generate1ItemSet(data)
    dataSet = convertToSet(data)
    while True:
        K += 1
        l = set()
        for a in L:
            for b in L:
                tmp = a | b
                tmp = sorted(tmp)
                if len(tmp) == K:
                    l.add(frozenset(tmp))
        L = []
        for a in l:
            count = 0
            for i in dataSet:
                if a.issubset(i):
                    count+=1
            if count/size >= min_sup:
                L.append(a)
        if len(L) == 0:
            break
        else:
            printUtil(len(L),K)


def generate1ItemSet(data, min_sup = 0.5):
    C = {}
    size = len(data)
    L = []
    for row in range(len(data)):
        for col in range(len(data[0])):
            if data[row][col] in C:
                C[data[row][col]] += 1
            else:
                C[data[row][col]] = 1
    count = 0
    for item in C:
        if C[item]/size > min_sup:
            tmp = set()
            tmp.add(item)
            L.append(tmp)
            count+=1
    printUtil(len(L), 1)
    return L

def convertToSet(dataList):
    dataSet = list()
    for row in dataList:
        tmp = set()
        for col in row:
            tmp.add(col)
        dataSet.append(tmp)
    return dataSet


def printUtil(count, K):
    print("Number of length {} frequent itemsets {}".format(K,count))
    
    

if __name__ == "__main__":
    data = read_data("../../Data/assrules.txt")
    frequentSetGeneration(data)