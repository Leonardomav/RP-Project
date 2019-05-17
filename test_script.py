import matplotlib
from test_pipeline import test_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import PCA
import pandas
from feature_selection_pipeline import kruskal_wallis, select_k_best, ROC
from prediction_pipeline import kfold_cross_val_predictions, train_test_predictions
import data_preprocessment

matplotlib.use('TkAgg')

data, states, num_columns, data_loc = data_preprocessment.get_preprocessed_data()

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
                        if classifier[0] == 'mahalanobis' and fit_transform_option is not None and fit_transform_option[
                            0] == 'lda-dr':
                            pass
                        elif fit_transform_option is not None:
                            test_pipeline(
                                data,
                                Pipeline([
                                    fit_transform_option,
                                    classifier
                                ]),
                                seed,
                                n_features=selected_feature,
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
