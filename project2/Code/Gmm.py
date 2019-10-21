import numpy as np
import pandas as pd

class gmm:

    def __init__(self, filePath):
        # mu = k x d, pi = 1 x k, sigma = k x d x d where k = number of clusters, d = dimensions/features
        self.mu = None
        self.pi = None
        self.sigma = None
        self.dataMatrix, geneIds = self.readData(filePath)
        self.reg_sigma = 1e-6*np.identity(len(self.dataMatrix[0]))
        self.readParams()
        self.emAlgorithm()

    def readParams(self):
        self.numClusters = int(input("enter number of clusters: "))
        self.threshold = int(input("enter convergence threshold: "))
        self.maxIterations = int(input("enter maximum iterations: "))
        self.mu = np.random.randint(min(self.dataMatrix[:,0]), max(self.dataMatrix[:,0]), 
                                                size=(self.numClusters, len(self.dataMatrix[0])))
        self.sigma = np.zeros((self.numClusters, len(self.dataMatrix[0]), len(self.dataMatrix[0])))
        for dim in range(len(self.sigma)):
            np.fill_diagonal(self.sigma[dim], 5)
        self.pi = np.ones(self.numClusters) / self.numClusters

    def readData(self, filePath):
        #Read data from text file as numpy ndarray
        data = np.genfromtxt(filePath, dtype='float', delimiter="\t")
        data = np.delete(data, [0,1], axis=1)
        geneIds = np.genfromtxt(filePath, delimiter="\t", dtype=str, usecols=0)
        return data, geneIds

    def emAlgorithm(self):
        for i in range(self.maxIterations):
            probMatrix = self.eStep()
            self.mStep(probMatrix)

    def eStep(self):
        # probMatrix (rik) = n x k where n = number of data points, k = number of clusters
        probMatrix = np.zeros((self.dataMatrix.shape[0], self.numClusters))
        for index in range(0, len(self.dataMatrix)):
            for mu, sig, pi, idx in zip(self.mu, self.sigma, self.pi, range(self.numClusters)):
                probMatrix[index, idx] = self.rik(self.dataMatrix[index].T, mu.T, sig, pi)
        return probMatrix

    def rik(self, x, mu, sigma, pi):
        numerator = pi*self.multivariateGaussian(x, mu, sigma+self.reg_sigma)
        denominator = np.sum([p*self.multivariateGaussian(x, m, s) 
                                    for p, m, s in zip(self.pi, self.mu, self.sigma+self.reg_sigma)])
        return numerator / denominator

    def multivariateGaussian(self, x, mu, sigma):
        # x = d x 1, mu = d x 1, sigma = d x d where d = dimensions/features of data
        xMean = x - mu
        sigma_inv = np.linalg.inv(sigma)
        return (1. / (np.sqrt((2 * np.pi)**sigma.shape[0] * np.linalg.det(sigma))) * 
            np.exp(-(xMean.T @ sigma_inv @ xMean) / 2))

    def mStep(self, probMatrix):
        self.mu = []
        self.sigma = []
        self.pi = []
        for k in range(len(probMatrix[0])):
            sigma_rik = np.sum(probMatrix[:,k], axis=0)
            x = probMatrix[:,k].reshape(1, len(probMatrix))
            new_mu = (1/sigma_rik)*np.sum(probMatrix[:,k].reshape(len(probMatrix), 1)*self.dataMatrix, axis=0)
            self.mu.append(new_mu)
            new_pi = sigma_rik / len(self.dataMatrix)
            self.pi.append(new_pi)
            new_sigma = (np.dot((np.array(probMatrix[:,k]).reshape(len(self.dataMatrix), 1)
                            *(self.dataMatrix - new_mu)).T, (self.dataMatrix - new_mu))) / sigma_rik
            self.sigma.append(new_sigma + self.reg_sigma)

        log_likelihood = np.sum([np.sum([probMatrix[i][k]*(np.log(p) + 
                            np.log(self.multivariateGaussian(self.dataMatrix[i], m, s))) for p, m, s, k in 
                            zip(self.pi, self.mu, self.sigma, range(self.numClusters))])
                            for i in range(self.numClusters)])
        print(log_likelihood)

if __name__ == "__main__":
    filename = input("enter file name (without extension): ")
    filePath = "CSE-601/project2/Data/"+ filename + ".txt"
    mm = gmm(filePath)