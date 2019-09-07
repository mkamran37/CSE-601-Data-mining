import numpy as np
def pca(matrix):
    matrix = np.array([[19.0,63.0],[39.0,74.0],[30.0,87.0],[30.0,23.0],[15.0,35.0],[15.0,43.0],[15.0,32.0],[30.0,73.0]])
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
    for col in range(matrix.shape[0]):
        result[col][0] = matrix[col][0]*PC1[0] + matrix[col][1]*PC1[1]
    for col in range(matrix.shape[1]):
        result[col][1] = matrix[col][0]*PC2[0] + matrix[col][1]*PC2[1]
    print(result)
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

if __name__ == "__main__":
    pca()