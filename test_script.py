import copy

import matplotlib
import pandas
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import data_preprocessment
from feature_selection_pipeline import ROC, kruskal_wallis, select_k_best
from prediction_pipeline import kfold_cross_val_predictions, train_test_predictions
from test_pipeline import test_pipeline

matplotlib.use('TkAgg')

data, states, num_columns, data_loc = data_preprocessment.get_preprocessed_data()

fit_transform_options = [
    None,
    #('lda-dr', LinearDiscriminantAnalysis()),
    #('pca', PCA()),
    #('kernel_density', KernelDensity(kernel='gaussian')),
    #('knearest_neighbours', KNeighborsClassifier()),
]

classifiers = [
    ('lda', LinearDiscriminantAnalysis()),
    ('euclidean', NearestCentroid(metric='euclidean')),
    ('mahalanobis', NearestCentroid(metric='mahalanobis')),
    ('bayes', GaussianNB()),
    ('knearest_neighbours', KNeighborsClassifier()),
    ('svm', SVC()),
]

feature_selection_functions = [
    #None,
    kruskal_wallis,
    select_k_best,
    ROC,
]

prediction_functions = [
    kfold_cross_val_predictions,
    #train_test_predictions,
]

selected_features = [
    #3,
    5,
    #10,
    #16,
]

seeds_to_test = 1

for fit_transform_option in fit_transform_options:
    for classifier in classifiers:
        for feature_selection_function in feature_selection_functions:
            for prediction_function in prediction_functions:
                for n_feature in selected_features:
                    for seed in range(seeds_to_test):
                        if classifier[0] == 'mahalanobis' and fit_transform_option is not None and fit_transform_option[
                            0] == 'lda-dr':
                            pass
                        elif fit_transform_option is not None:
                            test_pipeline(
                                copy.deepcopy(data),
                                Pipeline([
                                    fit_transform_option,
                                    classifier
                                ]),
                                seed,
                                n_features=n_feature,
                                feature_selection_function=feature_selection_function,
                                prediction_function=prediction_function)
                        else:
                            test_pipeline(
                                copy.deepcopy(data),
                                Pipeline([
                                    classifier
                                ]),
                                seed,
                                n_features=n_feature,
                                feature_selection_function=feature_selection_function,
                                prediction_function=prediction_function)
