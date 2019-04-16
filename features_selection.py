import scipy
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import pandas as pd


def kruskal_wallis(data, data_y):
    rank = []
    setY = data_y['RainTomorrow'].tolist()
    for i in range(data.shape[1]):
        setX = data[data.columns[i]].tolist()
        rank.append((i, scipy.stats.kruskal(setX, setY)[0]))

    rank = sorted(rank, key=lambda x: x[1])
    return rank


def select_k_best(data, data_y, n_col=16):
    clf = SelectKBest(mutual_info_classif, k=n_col)
    new_data = clf.fit_transform(data,data_y)
    mask = clf.get_support()
    cols = data.columns[mask]
    new_data=pd.DataFrame(data=new_data, columns=cols)
    return new_data
