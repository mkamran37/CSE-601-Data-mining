import numpy as np
from pca import read_data, scatter_plot

def svd(data):
    U, D, V = np.linalg.svd(data)
    return U

if __name__ == "__main__":
    filename = input("enter file name (without extension)")
    data, diseases = read_data("../../Data/"+filename+".txt")
    svd_result = svd(data)
    scatter_plot(svd_result, diseases, filename, "SVD")