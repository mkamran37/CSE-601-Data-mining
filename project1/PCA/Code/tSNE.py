from sklearn.manifold import TSNE
import numpy as np
from pca import read_data

def T_SNE(data):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    tsne_results = tsne.fit_transform(data)
    print(tsne_results)

if __name__ == "__main__":
    data, diseases = read_data("../../Data/pca_a.txt")
    # data, diseases = read_data("CSE-601/project1/Data/pca_b.txt")
    # data, diseases = read_data("CSE-601/project1/Data/pca_c.txt")
    T_SNE(data)
