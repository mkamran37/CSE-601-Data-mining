#ALGORITHM:
#Ci is the candidate set, min_sup = minimum support, min_conf = minimum confidence
#Li is the set of candidates whose support >=min_sup
#K = 1
#Step1: Generate all the 1-itemset along with their support
#Step2: Remove all itemsets whose support<min_sup
#Step3: K=K+1
#repeat the above steps till |Lk| == 0;
import numpy as np
import itertools as it

def read_data(filepath):
    #Read data from text file as numpy ndarray
    data = np.genfromtxt(filepath, dtype='str', delimiter="\t")
    for i in range(data.shape[0]):
        t = 1
        for j in range(data.shape[1] - 1):
            data[i][j] = 'G'+str(t)+'_'+data[i][j]
            t+=1
    return np.ndarray.tolist(data)

def frequentSetGeneration(data, min_sup = 50):
    min_sup /=100
    K = 1
    L = []
    result = dict()
    size = len(data)
    L, dictionary = generate1ItemSet(data, min_sup)
    result = dictionary
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
                result[a] = count
        if len(L) == 0:
            break
        else:
            printUtil(L,len(L),K)
    return result


def generate1ItemSet(data, min_sup):
    C = {}
    size = len(data)
    L = []
    dictionary = dict()
    for row in range(len(data)):
        for col in range(len(data[row])):
            if data[row][col] in C:
                C[data[row][col]] += 1
            else:
                C[data[row][col]] = 1
    count = 0
    for item in C:
        if C[item]/size >= min_sup:
            tmp = set()
            tmp.add(item)
            dictionary[frozenset(tmp)] = C[item]
            L.append(tmp)
            count+=1
    printUtil(L,len(L), 1)
    return L, dictionary

def convertToSet(dataList):
    dataSet = list()
    for row in dataList:
        tmp = set()
        for col in row:
            tmp.add(col)
        dataSet.append(tmp)
    return dataSet

def printUtil(L, count, K):
    print("Number of length {} frequent itemsets {}".format(K,count))
    f = open("length"+str(K)+"output.txt", "a+")
    for i in L:
        f.write(str(set(i))+'\n')
    

if __name__ == "__main__":
    filename = input("enter file name (without extension)")
    data = read_data("../../Data/"+filename+".txt")
    min_sup = float(input("Enter minimum support (in %): "))
    frequentSetGeneration(data, min_sup)     