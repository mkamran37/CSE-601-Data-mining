import numpy as np

class k_means:
    def __init__(self):
        filename = input("enter file name (without extension)")
        self.read_data("../Data/"+filename+".txt")
    
    def read_data(self, filepath):
        #Read data from text file as numpy ndarray
        data = np.genfromtxt(filepath, dtype='double', delimiter="\t")
        dataset = dict()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if j == 0:
                    dataset[int(data[i][j])] = list()
                elif j == 1:
                    continue
                else:
                    dataset.get(data[i][0]).append(data[i][j])
    
    def find_cluster(self, point, clusters):
        

        

if __name__ == "__main__":
    kmeans = k_means()


            