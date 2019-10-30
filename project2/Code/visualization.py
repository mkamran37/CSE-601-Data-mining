from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class visualization:

    def pca(self, dataset, result, ids):
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(dataset)
        # r = np.array(result)
        pca_matrix = pca.transform(dataset)
        pca_matrix_df = pd.DataFrame(pca_matrix, columns=['PC1', 'PC2'], index=ids)
        # df = pd.DataFrame(data = np.concatenate((pca_matrix, result.to_numpy()), axis = 1), columns=['PC1','PC2','Cluster'])
        df = pd.concat([pca_matrix_df, result], axis=1)
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        plt.show()