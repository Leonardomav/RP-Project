from sklearn.neighbors.nearest_centroid import NearestCentroid

def Euclidian_MDC(X_train, X_test, y_train, y_test):
    clf = NearestCentroid(metric='euclidean')
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(clf.score(X_test, y_test))

def Mahalanobis_MDC():
    pass