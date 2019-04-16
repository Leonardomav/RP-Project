import matplotlib
matplotlib.use('TkAgg')
from scipy.stats import kstest
from test_pipeline import test_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import PCA
import pandas
from features_selection import kruskal_wallis

def categorize_data(data):
    labels = data['WindGustDir'].astype('category').cat.categories.tolist()
    col = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

    for c in col:
        replace_map = {c: {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
        data.replace(replace_map, inplace=True)

    labels = data['Location'].astype('category').cat.categories.tolist()
    replace_map = {'Location': {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
    data.replace(replace_map, inplace=True)
    data['Location'].astype('category')

    return data

def get_preprocessed_data():
    # Load data set
    filename = 'weatherAUS.csv'
    data_raw = pandas.read_csv(filename)

    # Remove features that have more than 20% of missing values
    data_less_raw = data_raw.dropna(1, thresh=len(data_raw.index) * 0.8)

    # Remove examples that have any missing values
    data_less_raw = data_less_raw.dropna(0, how='any')

    # Remove RISK_MM
    data_less_raw = data_less_raw.drop(['RISK_MM'], axis=1)

    data = data_less_raw.copy()
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})

    data = categorize_data(data)

    data_y = data['RainTomorrow'].ravel()
    data = data.drop(['Date', 'Location'], axis=1)

    return {'x': data, 'y': data_y}

def kfold_cross_val_predictions(pipeline, data, seed):
    kfold = KFold(n_splits=10, random_state=seed)
    predictions = cross_val_predict(pipeline, data['x'], data['y'], cv=kfold)
    return data['y'], predictions, "kfold_cross_val_predictions"

def train_test_predictions(pipeline, data, seed):
    X_train, X_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=0.25, random_state=seed)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return y_test, predictions, "train_test_predictions"


data = get_preprocessed_data()

logistic = SGDClassifier(loss='log', penalty='l2', early_stopping=True,
                         max_iter=10000, tol=1e-5, random_state=0)

ass = SelectKBest(mutual_info_classif)
ress = ass.fit_transform(data['x'], data['y'])

ass2 = kruskal_wallis(data['x'], data['y'])
ass3 =SelectKBest(chi2) 
ress = ass3.fit_transform(data['x'], data['y'])
test_pipeline(
    data,
    Pipeline([
        ('lda', LinearDiscriminantAnalysis()),
        ]),
    0,
    kfold_cross_val_predictions)
test_pipeline(
    data,
    Pipeline([
        ('pca', PCA()),
        ('lda', LinearDiscriminantAnalysis()),
        ]),
    0,
    kfold_cross_val_predictions)
test_pipeline(
    data,
    Pipeline([
        ('lda-dr', LinearDiscriminantAnalysis()),
        ('lda', LinearDiscriminantAnalysis()),
        ]),
    0,
    kfold_cross_val_predictions)

test_pipeline(
    data,
    Pipeline([
        ('nearest_centroid', NearestCentroid()),
        ]),
    0,
    kfold_cross_val_predictions)
test_pipeline(
    data,
    Pipeline([
        ('pca', PCA()),
        ('nearest_centroid', NearestCentroid()),
        ]),
    0,
    kfold_cross_val_predictions)
test_pipeline(
    data,
    Pipeline([
        ('lda-dr', LinearDiscriminantAnalysis()),
        ('nearest_centroid', NearestCentroid()),
        ]),
    0,
    kfold_cross_val_predictions)
test_pipeline(
    data,
    Pipeline([
        ('lda', LinearDiscriminantAnalysis()),
        ]),
    0,
    train_test_predictions)
test_pipeline(
    data,
    Pipeline([
        ('pca', PCA()),
        ('lda', LinearDiscriminantAnalysis()),
        ]),
    0,
    train_test_predictions)
test_pipeline(
    data,
    Pipeline([
        ('lda-dr', LinearDiscriminantAnalysis()),
        ('lda', LinearDiscriminantAnalysis()),
        ]),
    0,
    train_test_predictions)

test_pipeline(
    data,
    Pipeline([
        ('nearest_centroid', NearestCentroid()),
        ]),
    0,
    train_test_predictions)
test_pipeline(
    data,
    Pipeline([
        ('pca', PCA()),
        ('nearest_centroid', NearestCentroid()),
        ]),
    0,
    train_test_predictions)
test_pipeline(
    data,
    Pipeline([
        ('lda-dr', LinearDiscriminantAnalysis()),
        ('nearest_centroid', NearestCentroid()),
        ]),
    0,
    train_test_predictions)

test_pipeline(
    data,
    Pipeline([
        ('kruskall-wallis', SelectKBest()),
        #('lda-dr', LinearDiscriminantAnalysis()),
        ('nearest_centroid', NearestCentroid()),
        ]),
    0,
    train_test_predictions)