import numpy as np

class k_means:
    def init(self):
        filename = input("enter file name (without extension)")
        data = read_data("../Data/"+filename+".txt")
    def read_data(filepath):
        #Read data from text file as numpy ndarray
        data = np.genfromtxt(filepath, dtype='int', delimiter="\t")
        dataset = dict()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if j == 0:
                    dataset[data[i][j]] = list()
                elif j == 1:
                    continue
                else:
                    dataset.get(data[i][0]).append(data[i][j])
        
        print(dataset)

if __name__ == "__main__":
    k_means = init()


            
