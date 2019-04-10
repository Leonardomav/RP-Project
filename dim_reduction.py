import sklearn.decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def PCA(data_y, data):
    x = data.loc[:, ~data.columns.isin(['Date', 'Location', 'RainTomorrow'])].values
    x_scaled = sklearn.preprocessing.StandardScaler().fit_transform(x)

    pca = sklearn.decomposition.PCA(n_components=2)
    principal_components = pca.fit(x_scaled).transform(x_scaled)

    data_y = data_y.iloc[:, 0].as_matrix()

    plt.figure()
    target_names = [0, 1]
    colors = ['r', 'b']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], target_names):
        print(principal_components[data_y == i, 0])
        plt.scatter(principal_components[data_y == i, 0], principal_components[data_y == i, 1], color=color, alpha=.8,
                    lw=lw, label=target_name, s=0.25)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of WAUS dataset')
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
