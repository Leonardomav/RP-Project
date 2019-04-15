import scipy
import sklearn.feature_selection


def kruskal_wallis(data, data_y):
    rank = []
    setY = data_y['RainTomorrow'].tolist()
    for i in range(data.shape[1]):
        setX = data[data.columns[i]].tolist()
        rank.append((i, scipy.stats.kruskal(setX, setY)[0]))

    rank = sorted(rank, key=lambda x: x[1])
    return rank


def select_k_best(data, data_y, n_col=16):
    new_data = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=n_col).fit_transform(data,
                                                                                                            data_y)
    return new_data
