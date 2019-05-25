import scipy
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.model_selection
from sklearn.metrics import roc_auc_score
from sklearn.neighbors.kde import KernelDensity
from sklearn.svm import SVR

import numpy as np


def kruskal_wallis(data, n_features=16, seed=None):
    """
    preforms feature selection using kruskal wallis
    Recieves:
        data -> data frame
        n_features -> number of remaining features
    Returns:
        keep_features -> list with the naime of the choosen features
    """

    rank = []
    for i in data['x'].columns:
        setX = data['x'][i].values
        rank.append((i, scipy.stats.kruskal(setX, data['y'])[0]))

    rank = sorted(rank, key=lambda x: x[1])
    keep_features = []
    for i in range(n_features):
        keep_features.append(rank[i][0])

    return keep_features


def select_k_best(data, n_features=16, seed=None):
    """
    Preforms feature selection using the select_k_best with mutual_info_classif
    Recieves:
        data -> data frame
        n_features -> number of remaining features
    Returns:
        keep_features -> list with the naime of the choosen features
    """
    clf = SelectKBest(mutual_info_classif, k=n_features)
    new_data = clf.fit_transform(data['x'], data['y'].ravel())
    mask = clf.get_support()
    keep_features = data['x'].columns[mask]

    return keep_features


def ROC(data, n_features=16, seed=None):
    """
    Preforms feature selection using the ROC with LDA classifier
    Recieves:
        data -> data frame
        n_features -> number of remaining features
    Returns:
        keep_features -> list with the naime of the choosen features
    """
    rank = []
    classifier = LinearDiscriminantAnalysis()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data['x'], data['y'], test_size=0.25,
                                                                                random_state=seed, shuffle=False)
    for i in X_train.columns:
        x_train_2d_array = X_train[i].to_frame()
        y_score = classifier.fit(x_train_2d_array, y_train)
        x_test_2d_array = X_test[i].to_frame()
        y_score = y_score.decision_function(x_test_2d_array)
        rank.append([i, roc_auc_score(y_test, y_score)])

    rank = sorted(rank, key=lambda x: x[1])
    keep_features = []
    for i in range(n_features):
        keep_features.append(rank[i][0])
    return keep_features


def kernel_density_fs(data, n_features=16, seed=None):
    """
    Preforms feature selection using the KernelDensity with gaussian kernel
    Recieves:
        data -> data frame
        n_features -> number of remaining features
    Returns:
        keep_features -> list with the naime of the choosen features
    """
    rank = []
    for column in data['x']:
        df = data['x'][column].values.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(df)

        values = kde.score(df, data['y'])
        rank.append((column, values))

    rank = sorted(rank, key=lambda x: x[1], reverse=True)
    keep_features = []
    for i in range(n_features):
        keep_features.append(rank[i][0])

    return keep_features

def RFE_fs(data, n_features=16, seed=None):
    """
    Preforms feature selection using the RFE with LDA
    Recieves:
        data -> data frame
        n_features -> number of remaining features
    Returns:
        keep_features -> list with the naime of the choosen features
    """
    estimator = LinearDiscriminantAnalysis()
    selector = RFE(estimator, n_features)
    selector = selector.fit(data['x'], data['y'])
    mask = selector.get_support()
    keep_features = data['x'].columns[mask]
    return keep_features
