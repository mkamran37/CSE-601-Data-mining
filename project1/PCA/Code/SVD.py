import numpy as np
from pca import read_data, scatter_plot

def svd(data):
    U, D, V = np.linalg.svd(data)
    return U

if __name__ == "__main__":
    data, diseases = read_data("CSE-601/project1/Data/pca_a.txt")
    # data, diseases = read_data("CSE-601/project1/Data/pca_b.txt")
    # data, diseases = read_data("CSE-601/project1/Data/pca_c.txt")
    svd_result = svd(data)
    scatter_plot(svd_result, diseases)