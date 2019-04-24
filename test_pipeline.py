import csv
import datetime
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)


def test_pipeline(data, pipeline, seed, n_features=16, feature_selection_function = None, prediction_function = None, visual=False):
    test_name = str(datetime.datetime.now()) + ":"
    for procedure in pipeline.named_steps:
        test_name += procedure + ';'
    test_name = test_name[:-1] + "-" + str(seed)
    if feature_selection_function != None:
        test_name += '-' + feature_selection_function.__name__
    if prediction_function != None:
        test_name += '-' + prediction_function.__name__
    print("Running ", test_name)
    if feature_selection_function != None:
        foo = feature_selection_function(data, n_features, seed)
    tested_data, predictions, prediction_function_name = prediction_function(pipeline, data, 0)
    test_name += "-" + prediction_function_name
    conf_matrix = confusion_matrix(tested_data, predictions)
    class_report = classification_report(tested_data, predictions, output_dict=True)
    if visual:
        plot_confusion_matrix(tested_data, predictions, [0, 1],
                            normalize=False,
                            title=None,
                            cmap=plt.cm.Blues)
        plt.show()
    with open('results.csv', 'a') as result_csv:
        writer = csv.writer(result_csv)
        writer.writerow(
            [test_name, 
            class_report['0']['f1-score'],
            class_report['0']['precision'],
            class_report['0']['recall'],
            class_report['1']['f1-score'],
            class_report['1']['precision'],
            class_report['1']['recall'],
            accuracy_score(tested_data, predictions),
            ])
    print('Done')
