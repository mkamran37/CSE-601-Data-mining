import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class gmm:

    def __init__(self, filePath, centroids):
        # mu = k x d, pi = 1 x k, sigma = k x d x d where k = number of clusters, d = dimensions/features
        self.mu = centroids
        self.pi = None
        self.sigma = None
        self.log_likelihood = [0]
        print("\nRequire input parameters for gaussian mixture model ....................")
        self.dataMatrix, self.dataIds = self.readData(filePath)
        self.readParams()

    def readParams(self):
        self.numClusters = int(input("Enter number of clusters: "))
        self.threshold = float(input("Enter convergence threshold: "))
        self.maxIterations = int(input("Enter maximum iterations: "))
        self.smooth = 1e-9*np.ones(len(self.dataMatrix[0]))
        self.mu = [[0,0],[3,3],[0,4]]
        # self.sigma = np.zeros((self.numClusters, len(self.dataMatrix[0]), len(self.dataMatrix[0])), dtype='float')
        self.sigma = [[[1,0.4],[0.4,1]], [[1,0], [0,2]], [[0.4,0], [0,0.1]]]
        # for dim in range(len(self.sigma)):
            # np.fill_diagonal(self.sigma[dim], 1)
        # self.pi = np.ones(self.numClusters) / self.numClusters
        self.pi = [0.4, 0.4, 0.2]

    def readData(self, filePath):
        #Read data from text file as numpy ndarray
        data = np.genfromtxt(filePath, dtype='float', delimiter="\t")
        data = np.delete(data, [0,1], axis=1)
        dataIds = np.genfromtxt(filePath, delimiter="\t", dtype=str, usecols=0)
        return data, dataIds

    def emAlgorithm(self):
        print("\nRunning Gaussian Mixture Model ....................")
        for i in range(self.maxIterations):
            self.probMatrix = self.eStep()
            logL = self.mStep()
            self.log_likelihood.append(logL)
            print("Log Likelihood (Iteration: " + str(i+1) + "): " + str(logL))
            if abs(self.log_likelihood[-1] - self.log_likelihood[-2]) < self.threshold:
                print("Reached convergence threshold ....................")
                break

        predictedMatrix = self.getClusters()

        print("\nMean ..........")
        print(self.mu)
        print("\nCovariance matrix ..........")
        print(self.sigma)
        print("\nPrior cluster probabilities ..........")
        print(self.pi)
        return self.dataMatrix, predictedMatrix, self.dataIds

    def eStep(self):
        # probMatrix (rik) = n x k where n = number of data points, k = number of clusters
        probMatrix = np.zeros((self.dataMatrix.shape[0], self.numClusters))
        for mu, sig, pi, idx in zip(self.mu, self.sigma, self.pi, range(self.numClusters)):
                sig += self.smooth
                probMatrix[:, idx] = self.rik(mu, sig, pi)
        return probMatrix

    def rik(self, mu, sigma, pi):
        numerator = pi*multivariate_normal.pdf(self.dataMatrix, mean=mu, cov=sigma, allow_singular=True)
        denominator = np.sum([p*multivariate_normal.pdf(self.dataMatrix, mean=m, cov=s, allow_singular=True)
                                    for p, m, s in zip(self.pi, self.mu, self.sigma+self.smooth)], axis=0)
        return numerator / denominator

    def mStep(self):
        self.mu = []
        self.sigma = []
        self.pi = []
        for k in range(len(self.probMatrix[0])):
            sigma_rik = np.sum(self.probMatrix[:,k], axis=0)
            new_mu = (1/sigma_rik)*np.sum(self.probMatrix[:,k].reshape(len(self.probMatrix), 1)*self.dataMatrix, axis=0)
            self.mu.append(new_mu)
            new_pi = sigma_rik / len(self.dataMatrix)
            self.pi.append(new_pi)
            new_sigma = (np.dot((np.array(self.probMatrix[:,k]).reshape(len(self.dataMatrix), 1)
                            *(self.dataMatrix - new_mu)).T, (self.dataMatrix - new_mu))) / sigma_rik
            self.sigma.append(new_sigma + self.smooth)

        log_likelihood = np.log(np.sum([pi*multivariate_normal.pdf(self.dataMatrix, mean=self.mu[m], cov=self.sigma[s]+self.smooth, allow_singular=True) 
                                    for pi,m,s in zip(self.pi,range(len(self.mu)),range(len(self.sigma)))]))
        return log_likelihood

    def getClusters(self):
        maxIndices = np.argmax(self.probMatrix, axis=1) + 1
        predictedMatrix = pd.DataFrame(maxIndices, index=self.dataIds, columns=['Cluster'])
        return predictedMatrix