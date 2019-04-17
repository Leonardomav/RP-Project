from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict

def kfold_cross_val_predictions(pipeline, data, seed):
    kfold = KFold(n_splits=10, random_state=seed)
    predictions = cross_val_predict(pipeline, data['x'], data['y'], cv=kfold)
    return data['y'], predictions, "kfold_cross_val_predictions"


def train_test_predictions(pipeline, data, seed):
    X_train, X_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=0.25, random_state=seed)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return y_test, predictions, "train_test_predictions"