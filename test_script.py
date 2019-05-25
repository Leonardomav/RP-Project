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
from feature_selection_pipeline import ROC, kruskal_wallis, select_k_best, kernel_density_fs, RFE_fs
from prediction_pipeline import kfold_cross_val_predictions, train_test_predictions
from test_pipeline import test_pipeline
import visualizations

matplotlib.use('TkAgg')

data, states, num_columns, data_loc, state_dict = data_preprocessment.get_preprocessed_data()


# NSW - New South Wales
# VIC - Victoria
# QNL - Queensland
# SAU - South Australia
# TAS - Tasmania
# WAU - Western Australia
# NRT - Northen Territoty

states = [
    "All",
    #"NSW",
    #"VIC",
    #"QNL",
    #"SAU",
    #"TAS",
    #"WAU",
    #"NRT",
]

if "All" not in states:
    data = data_preprocessment.select_location(data_loc, states, state_dict)

fit_transform_options = [
    #None,
    ('lda-dr', LinearDiscriminantAnalysis()),
    ('pca', PCA()),
]

classifiers = [
    #('lda', LinearDiscriminantAnalysis()),
     ('euclidean', NearestCentroid(metric='euclidean')),
     ('mahalanobis', NearestCentroid(metric='mahalanobis')),
    #('bayes', GaussianNB()),
    #('knearest_neighbours', KNeighborsClassifier()),
    #('svm', SVC()),
]

feature_selection_functions = [
    None,
    # kruskal_wallis,
    #select_k_best,
    # kernel_density_fs
    # ROC,
    # RFE_fs,
]

prediction_functions = [
    kfold_cross_val_predictions,
    # train_test_predictions,
]

selected_features = [
    # 3,
    5,
    # 10,
    # 16,
]

seeds_to_test = 1

predictions_list = []
for seed in range(seeds_to_test):
    for fit_transform_option in fit_transform_options:
        for classifier in classifiers:
            for feature_selection_function in feature_selection_functions:
                for prediction_function in prediction_functions:
                    for n_feature in selected_features:
                        if fit_transform_option is not None and fit_transform_option[0] == 'pca':
                            fit_transform_option = ('pca', PCA(n_components=n_feature))
                        if classifier[0] == 'mahalanobis' and fit_transform_option is not None and fit_transform_option[
                            0] == 'lda-dr':
                            pass
                        elif fit_transform_option is not None:
                            predictions_list.append(test_pipeline(
                                copy.deepcopy(data),
                                Pipeline([
                                    fit_transform_option,
                                    classifier
                                ]),
                                seed,
                                n_features=n_feature,
                                feature_selection_function=feature_selection_function,
                                prediction_function=prediction_function).tolist())
                        else:
                            predictions_list.append(test_pipeline(
                                copy.deepcopy(data),
                                Pipeline([
                                    classifier
                                ]),
                                seed,
                                n_features=n_feature,
                                feature_selection_function=feature_selection_function,
                                prediction_function=prediction_function).tolist())

# visualizations.box_plot_comparison(predictions_list)
