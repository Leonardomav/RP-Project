import scipy
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.model_selection
from sklearn.metrics import roc_auc_score
import numpy as np


def kruskal_wallis(data, data_y,n_features=16):
    rank = []
    setY = data_y['RainTomorrow'].tolist()
    for i in range(data.shape[1]):
        setX = data[data.columns[i]].tolist()
        rank.append((i, scipy.stats.kruskal(setX, setY)[0]))

    rank = sorted(rank, key=lambda x: x[1])
    keep_index = []
    for i in range(n_features):
        keep_index.append(rank[i][0])

    return keep_index


def select_k_best(data, data_y, n_features=16):
    clf = SelectKBest(mutual_info_classif, k=n_features)
    new_data = clf.fit_transform(data, data_y.values.ravel())
    mask = clf.get_support()
    keep_index=np.where(mask)[0]
    return keep_index


def ROC(data, data_y, n_features=16):
    rank = []
    classifier=LinearDiscriminantAnalysis()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data_y, test_size=0.25, random_state=42, shuffle=False)
    for i in range(len(X_train.columns)):
        X = X_train.iloc[:,i].to_frame()
        y_score = classifier.fit(X, y_train)
        y_score = y_score.decision_function(X)
        rank.append([i,roc_auc_score(y_test, y_score.values.ravel())])

    rank = sorted(rank, key=lambda x: x[1])
    keep_index = []
    for i in range(n_features):
        keep_index.append(rank[i][0])

    return keep_index

