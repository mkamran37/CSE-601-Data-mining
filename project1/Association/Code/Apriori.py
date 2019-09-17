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
    return data

def frequentSetGeneration(data, min_sup = 0.5):
    K = 1
    C = {}
    L = []
    size = data.shape[0]
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if data[row][col] in C:
                C[data[row][col]] += 1
            else:
                C[data[row][col]] = 1
    tmp = []
    count = 0
    for i, item in enumerate(C):
        if C[item]/size > min_sup:
            tmp[i] = item
            count+=1
    print("Number of length 1 frequent itemsets "+count)
    L.append(tmp)
    K += 1
    C = {}
    for i in range(len(tmp) - 1):
        for j in range(i+1,len(tmp)):
            
    
    

if __name__ == "__main__":
    data = read_data("../../Data/assrules.txt")
    frequentSetGeneration(data)
    # print(data[0])