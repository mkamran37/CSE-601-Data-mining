import matplotlib.pyplot as plt
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

def pca(matrix):
    # matrix = np.array([[19.0,63.0],[39.0,74.0],[30.0,87.0],[30.0,23.0],[15.0,35.0],[15.0,43.0],[15.0,32.0],[30.0,73.0]])
    meanList = []
    n = matrix.shape[0]
    for col in range(matrix.shape[1]):
        sum = 0
        for row in range(matrix.shape[0]):
            sum+=matrix[row,col]
        meanList.append(sum/n)
    matrix = adjustment(matrix, meanList)
    S = (1/(n-1))*np.matmul(matrix.T,matrix)
    eigenValues, eigenVectors = np.linalg.eig(S)
    PC1, PC2 = top2(eigenValues, eigenVectors)
    return reduceDimensions(PC1, PC2, PC1.shape[0], matrix)

def reduceDimensions(PC1, PC2, n, matrix):
    w, h = 2, matrix.shape[0]
    result = [[0 for x in range(w)] for y in range(h)] 
    print(PC1)
    for col in range(matrix.shape[0]):
        result[col][0] = np.dot(matrix[col],PC1)
    for col in range(matrix.shape[0]):
        result[col][1] = np.dot(matrix[col],PC2)
    return result

def top2(eigenValues, eigenVectors):
    result1 = np.where(eigenValues == np.amax(eigenValues))
    eigenValues = np.delete(eigenValues, result1[0])
    result2 = np.where(eigenValues == np.amax(eigenValues))
    return eigenVectors[result1[0][0]],eigenVectors[result2[0][0]]

def adjustment(matrix, meanList):
    tmp = 0
    for col in range(matrix.shape[1]):
        for row in range(matrix.shape[0]):
            matrix[row,col] = matrix[row,col] - meanList[tmp]
        tmp+=1
    return matrix

#Scatter Plot
def scatter_plot(matrix, diseases):
    plt.scatter(np.array(matrix)[:,0], np.array(matrix)[:,1])
    # colorbar = plt.colorbar()
    plt.show()

if __name__ == "__main__":
    data, diseases = read_data("CSE-601/project1/Data/pca_a.txt")
    matrix = pca(data)
    print(np.array(matrix)[:,0])
    print(np.array(matrix)[:,1])
    scatter_plot(matrix, diseases)
    data, diseases = read_data("CSE-601/project1/Data/pca_b.txt")
    data, diseases = read_data("CSE-601/project1/Data/pca_c.txt")
