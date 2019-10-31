from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class visualization:

    def pca(self, dataset, result, ids):
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(dataset)
        pca_matrix = pca.transform(dataset)
        pca_matrix_df = pd.DataFrame(pca_matrix, columns=['PC1', 'PC2'], index=ids)
        df = pd.concat([pca_matrix_df, result], axis=1)
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        plt.show()
    
    def visualize(self, dataset, result, ids):
        pdf = pd.DataFrame(dataset, columns=['Col1','Col2'], index=ids)
        df = pd.concat([pdf, result], axis=1)
        lm = sns.lmplot(x='Col1', y='Col2', data=df, fit_reg=False, hue='Cluster')
        plt.show()
