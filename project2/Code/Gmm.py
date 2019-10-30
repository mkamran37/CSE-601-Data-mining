import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class gmm:

    def __init__(self, filePath, centroids):
        # mu = k x d, pi = 1 x k, sigma = k x d x d where k = number of clusters, d = dimensions/features
        self.mu = centroids
        self.pi = None
        self.sigma = None
        self.dataMatrix, self.geneIds = self.readData(filePath)
        self.reg_sigma = 1e-9*np.identity(len(self.dataMatrix[0]))
        self.readParams()

    def readParams(self):
        self.numClusters = int(input("enter number of clusters: "))
        self.threshold = int(input("enter convergence threshold: "))
        self.maxIterations = int(input("enter maximum iterations: "))
        self.mu = [[0,0],[1,1]]
        # self.sigma = np.zeros((self.numClusters, len(self.dataMatrix[0]), len(self.dataMatrix[0])), dtype='float')
        self.sigma = [[[1,1],[1,1]], [[2,2], [2,2]]]
        # for dim in range(len(self.sigma)):
        #     np.fill_diagonal(self.sigma[dim], 1)
        # self.pi = np.ones(self.numClusters) / self.numClusters
        self.pi = [0.5, 0.5]

    def readData(self, filePath):
        #Read data from text file as numpy ndarray
        data = np.genfromtxt(filePath, dtype='float', delimiter="\t")
        data = np.delete(data, [0,1], axis=1)
        geneIds = np.genfromtxt(filePath, delimiter="\t", dtype=str, usecols=0)
        return data, geneIds

    def emAlgorithm(self):
        for i in range(self.maxIterations):
            self.probMatrix = self.eStep()
            self.log_likelihood = self.mStep()
            print("Log Likelihood (Iteration: " + (i+1) + "): " + str(self.log_likelihood))

        predictedMatrix = self.getClusters()
        return self.dataMatrix, predictedMatrix, self.geneIds

    def eStep(self):
        # probMatrix (rik) = n x k where n = number of data points, k = number of clusters
        probMatrix = np.zeros((self.dataMatrix.shape[0], self.numClusters))
        for mu, sig, pi, idx in zip(self.mu, self.sigma, self.pi, range(self.numClusters)):
                probMatrix[:, idx] = self.rik(mu, sig, pi)
        return probMatrix

    def rik(self, mu, sigma, pi):
        numerator = pi*multivariate_normal(mean=mu, cov=sigma, allow_singular=True).pdf(self.dataMatrix)
        denominator = np.sum([p*multivariate_normal(mean=m, cov=s, allow_singular=True).pdf(self.dataMatrix)
                                    for p, m, s in zip(self.pi, self.mu, self.sigma+self.reg_sigma)], axis=0)
        return numerator / denominator

    def mStep(self):
        self.mu = []
        self.sigma = []
        self.pi = []
        for k in range(len(self.probMatrix[0])):
            sigma_rik = np.sum(self.probMatrix[:,k], axis=0)
            x = self.probMatrix[:,k].reshape(1, len(self.probMatrix))
            new_mu = (1/sigma_rik)*np.sum(self.probMatrix[:,k].reshape(len(self.probMatrix), 1)*self.dataMatrix, axis=0)
            self.mu.append(new_mu)
            new_pi = sigma_rik / len(self.dataMatrix)
            self.pi.append(new_pi)
            new_sigma = (np.dot((np.array(self.probMatrix[:,k]).reshape(len(self.dataMatrix), 1)
                            *(self.dataMatrix - new_mu)).T, (self.dataMatrix - new_mu))) / sigma_rik
            self.sigma.append(new_sigma + self.reg_sigma)

        log_likelihood = np.log(np.sum([pi*multivariate_normal(mean=self.mu[m],cov=self.sigma[s], allow_singular=True).pdf(self.dataMatrix) 
                                    for pi,m,s in zip(self.pi,range(len(self.mu)),range(len(self.sigma)))]))
        return log_likelihood

    def getClusters(self):
        maxIndices = np.argmax(self.probMatrix, axis=1) + 1
        predictedMatrix = pd.DataFrame(maxIndices, index=self.geneIds, columns=['Cluster'])
        return predictedMatrix