from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class visualization:
    def pca(self, datasett, result):
        pca = PCA(n_components=2, svd_solver='full')
        dataset = [data.point for data in datasett]
        pca.fit(dataset)
        r = np.array(result)
        pca_matrix = pca.transform(dataset)
        df = pd.DataFrame(data = np.concatenate((pca_matrix, r[:,1:2]), axis = 1), columns=['PC1','PC2','Cluster'])
        lm = sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='Cluster')
        plt.show()