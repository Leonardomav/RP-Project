from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from statistics import mean


def kfold_cross_val_predictions(pipeline, data, seed):
    """
    Preforms KFold prediction with cross validation from cross_val_predict
    Recieves:
        pipeline -> test pipeline
        data -> dataframe to predict
    Returns:
        data['y'] -> real values
        predictions -> predicted values
    """
    kfold = KFold(n_splits=10, random_state=seed)
    predictions = cross_val_predict(pipeline, data['x'], data['y'], cv=kfold)
    results = cross_val_score(pipeline,  data['x'], data['y'], cv=kfold, scoring='accuracy')
    return data['y'], predictions, results


def train_test_predictions(pipeline, data, seed):
    """
    Preforms train_test_split to split data and fits data to the model and predicts with no validation
    Recieves:
        pipeline -> test pipeline
        data -> dataframe to predict
    Returns:
        y_test -> real values
        predictions -> predicted values
    """
    X_train, X_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=0.25, random_state=seed)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return y_test, predictions, []


