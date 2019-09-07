import numpy as np

#Read files and get data points
#input : .txt file with n rows and d columns
#output : an n*d matrix
def read_data(filepath):
    #Read data from text file as numpy ndarray
    data = np.genfromtxt(filepath, delimiter="\t")
    #Removed the last column(diseases) from data ndarray
    data = np.delete(data, -1, axis=1)
    #Read last column(diseases) into a seperate array
    diseases = np.genfromtxt(filepath, delimiter="\t", dtype=str, usecols=-1)
    return data, diseases

#PCA Algorithm
#input: n*d matrix
#output: 2*2 matrix


data, diseases = read_data("CSE-601/project1/Data/pca_a.txt")
data, diseases = read_data("CSE-601/project1/Data/pca_b.txt")
data, diseases = read_data("CSE-601/project1/Data/pca_c.txt")