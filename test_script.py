import matplotlib
matplotlib.use('TkAgg')
from scipy.stats import kstest
from test_pipeline import test_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import PCA
import pandas
from feature_selection_pipeline import kruskal_wallis, select_k_best, ROC
from prediction_pipeline import kfold_cross_val_predictions, train_test_predictions


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
    data = data.drop(['Date', 'Location', 'RainTomorrow'], axis=1)

    return {'x': data, 'y': data_y}

data = get_preprocessed_data()

fit_transform_options = [
    None,
    ('lda-dr', LinearDiscriminantAnalysis()),
    ('pca', PCA()),
]

classifiers = [
    ('lda', LinearDiscriminantAnalysis()),
    ('euclidean', NearestCentroid(metric='euclidean')),
    ('mahalanobis', NearestCentroid(metric='mahalanobis')),
]

feature_selection_functions = [
    None,
    kruskal_wallis,
    select_k_best,
    ROC,
]

prediction_functions = [
    kfold_cross_val_predictions,
    train_test_predictions,
]

selected_features = [
    3,
    5,
    10,
]

seeds_to_test = 1

for fit_transform_option in fit_transform_options:
    for classifier in classifiers:
        for feature_selection_function in feature_selection_functions:
            for prediction_function in prediction_functions:
                for selected_feature in selected_features:
                    for seed in range(seeds_to_test):
                        if classifier[0] == 'mahalanobis' and fit_transform_option != None and fit_transform_option[0] =='lda-dr':
                            pass
                        elif fit_transform_option != None:
                            test_pipeline(
                                data,
                                Pipeline([
                                    fit_transform_option,                            
                                    classifier
                                ]),
                                seed,
                                n_features=selected_features,
                                feature_selection_function=feature_selection_function,
                                prediction_function=prediction_function)
                        else:
                            test_pipeline(
                                data,
                                Pipeline([
                                    classifier
                                ]),
                                seed,
                                n_features=selected_feature,
                                feature_selection_function=feature_selection_function,
                                prediction_function=prediction_function) 