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
predictions_list.append(test_pipeline(
    copy.deepcopy(data),
    Pipeline([
    ('lda-dr', LinearDiscriminantAnalysis()),
    ('euclidean', NearestCentroid(metric='euclidean')),
    ]),
    1,
    n_features=5,
    feature_selection_function=None,
    prediction_function=kfold_cross_val_predictions).tolist())

predictions_list.append(test_pipeline(
    copy.deepcopy(data),
    Pipeline([
    ('lda-dr', LinearDiscriminantAnalysis()),
    ('lda', LinearDiscriminantAnalysis()),
    ]),
    1,
    n_features=5,
    feature_selection_function=None,
    prediction_function=kfold_cross_val_predictions).tolist())

predictions_list.append(test_pipeline(
    copy.deepcopy(data),
    Pipeline([
    ('lda', LinearDiscriminantAnalysis()),
    ]),
    1,
    n_features=5,
    feature_selection_function=RFE_fs,
    prediction_function=kfold_cross_val_predictions).tolist())

predictions_list.append(test_pipeline(
    copy.deepcopy(data),
    Pipeline([
     ('mahalanobis', NearestCentroid(metric='mahalanobis')),
    ]),
    1,
    n_features=5,
    feature_selection_function=RFE_fs,
    prediction_function=kfold_cross_val_predictions).tolist())

visualizations.box_plot_comparison(predictions_list, ["lda-dr euclidean", "lda-dr lda", "rfe lda", "rfe mahalanobis"])
