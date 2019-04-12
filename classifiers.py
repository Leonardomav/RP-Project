from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import DistanceMetric
import numpy as np

def Euclidian_MDC(X_train, X_test, y_train, y_test):
    clf = NearestCentroid(metric='euclidean')
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

def Mahalanobis_MDC(X_train, X_test, y_train, y_test):
    dist = DistanceMetric.get_metric('mahalanobis', V=np.cov(X_train.values))
    clf = NearestCentroid(metric=dist)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

