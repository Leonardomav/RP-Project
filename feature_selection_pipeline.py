import scipy
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.model_selection
from sklearn.metrics import roc_auc_score
import numpy as np


def kruskal_wallis(data, n_features=16, seed=None):
    rank = []
    for i in data['x'].columns:
        setX = data['x'][i].values
        rank.append((i, scipy.stats.kruskal(setX, data['y'])[0]))

    rank = sorted(rank, key=lambda x: x[1])
    keep_index = []
    for i in range(n_features):
        keep_index.append(rank[i][0])

    return keep_index


def select_k_best(data, n_features=16, seed=None):
    clf = SelectKBest(mutual_info_classif, k=n_features)
    new_data = clf.fit_transform(data['x'], data['y'].ravel())
    mask = clf.get_support()
    keep_index=np.where(mask)[0]
    return keep_index


def ROC(data, n_features=16, seed=None):
    rank = []
    classifier=LinearDiscriminantAnalysis()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data['x'], data['y'], test_size=0.25, random_state=seed, shuffle=False)
    for i in X_train.columns:
        x_train_2d_array = X_train[i].to_frame()
        y_score = classifier.fit(x_train_2d_array, y_train)
        x_test_2d_array = X_test[i].to_frame()
        y_score = y_score.decision_function(x_test_2d_array)
        rank.append([i,roc_auc_score(y_test, y_score)])

    rank = sorted(rank, key=lambda x: x[1])
    keep_index = []
    for i in range(n_features):
        keep_index.append(rank[i][0])

    return keep_index

