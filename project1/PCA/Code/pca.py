import matplotlib.pyplot as plt

#Read files and get data points
#input : .txt file with n rows and d columns
#output : an n*d matrix

def read_data():
    #TODO

#PCA Algorithm
#input: n*d matrix
#output: 2*2 matrix
def pca(matrix):
    #TODO

#Scatter Plot
def scatter_plot(data, diseases):
    plt.scatter(data[:,0], data[:,1], c=diseases)
    colorbar = plt.colorbar()
    plt.show()
