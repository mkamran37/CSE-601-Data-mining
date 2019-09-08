import numpy as np

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
    PC1, PC2 = topN(eigenValues, eigenVectors)
    print(PC1, PC2)
    return reduceDimensions(PC1, PC2, PC1.shape[0], matrix)

def reduceDimensions(PC1, PC2, n, matrix):
    w, h = 2, matrix.shape[0]
    result = [[0 for x in range(w)] for y in range(h)] 
    for col in range(matrix.shape[0]):
        result[col][0] = np.dot(matrix[col],PC1)
    for col in range(matrix.shape[0]):
        result[col][1] = np.dot(matrix[col],PC2)
    return result

def topN(eigenValues, eigenVectors,n = 2):
    indices = eigenValues.argsort()[::-1][:n]
    return eigenVectors[indices[0]],eigenVectors[indices[1]]

def adjustment(matrix, meanList):
    tmp = 0
    for col in range(matrix.shape[1]):
        for row in range(matrix.shape[0]):
            matrix[row,col] = matrix[row,col] - meanList[tmp]
        tmp+=1
    return matrix

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


if __name__ == "__main__":
    data, diseases = read_data("../../Data/pca_a.txt")
    # data, diseases = read_data("CSE-601/project1/Data/pca_b.txt")
    # data, diseases = read_data("CSE-601/project1/Data/pca_c.txt")
    pca(data)

