from sklearn.manifold import TSNE
from pca import read_data, scatter_plot

def T_SNE(data):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    return tsne.fit_transform(data)

if __name__ == "__main__":
    data, diseases = read_data("../../Data/pca_a.txt")
    tsne_result = T_SNE(data)
    scatter_plot(tsne_result, diseases, "pca_a", "t-SNE")
    data, diseases = read_data("../../Data/pca_b.txt")
    tsne_result = T_SNE(data)
    scatter_plot(tsne_result, diseases, "pca_b", "t-SNE")
    data, diseases = read_data("../../Data/pca_c.txt")
    tsne_result = T_SNE(data)
    scatter_plot(tsne_result, diseases, "pca_c", "t-SNE")
