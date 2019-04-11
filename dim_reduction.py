import sklearn.decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import pandas as pd


def PCA(data_y, data):
    x = data.loc[:, ~data.columns.isin(['Date', 'Location', 'RainTomorrow'])].values
    x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(x)

    n_comp=2
    pca = sklearn.decomposition.PCA(n_components=n_comp)
    principal_components = pca.fit(x_scaled).transform(x_scaled)

    data_y = data_y.iloc[:, 0].as_matrix() #list do darray

    plt.figure()
    target_names = [0, 1]
    colors = ['r', 'b']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(principal_components[data_y == i, 0], principal_components[data_y == i, 1], color=color, alpha=.8,
                    lw=lw, label=target_name, s=0.25)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of WAUS dataset')
    plt.show()

def get_data_PCA(principal_components, data_y, n_comp):

    for i in range(n_comp):


    data_PCA = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    data_PCA = pd.concat([data_PCA, data_y[['RainTomorrow']]], axis=1)

    return data_PCA



def variance_feature_PCA(data):
    x = data.loc[:, ~data.columns.isin(['Date', 'Location', 'RainTomorrow'])].values
    x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(x)

    pca = sklearn.decomposition.PCA(n_components=4)
    pca.fit(x_scaled)

    plt.figure(1, figsize=(9, 8))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('Number of Feautres')
    plt.ylabel('Variance Ratio')
    plt.show()


def LDA(data_y, data):
    x = data.loc[:, ~data.columns.isin(['Date', 'Location', 'RainTomorrow'])].values
    x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(x)

    data_y = data_y.iloc[:, 0].as_matrix()

    lda = LinearDiscriminantAnalysis(n_components=2)
    principal_components = lda.fit_transform(x_scaled, data_y)

    plt.figure()
    target_names = [0, 1]
    colors = ['r', 'b']

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(principal_components[data_y == i, 0], principal_components[data_y == i, 0], alpha=.2, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of WAUS dataset')
    plt.show()

