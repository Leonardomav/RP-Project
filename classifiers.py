from sklearn.neighbors.nearest_centroid import NearestCentroid


def Euclidean_MDC(X_train, X_test, y_train, y_test):
    clf = NearestCentroid(metric='euclidean')
    clf.fit(X_train, y_train.values.ravel())
    print(clf.score(X_test, y_test))

def Mahalanobis_MDC(X_train, X_test, y_train, y_test):
    clf = NearestCentroid(metric='mahalanobis')
    clf.fit(X_train, y_train.values.ravel())
    print(clf.score(X_test, y_test))

