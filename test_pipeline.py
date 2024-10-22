import csv
import datetime
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from visualizations import plot_confusion_matrix
from prediction_pipeline import kfold_cross_val_predictions


def test_pipeline(data, pipeline, seed, n_features=16, feature_selection_function=None, prediction_function=None,
                  visual=False, region='All'):
    csv_values = [str(datetime.datetime.now())]
    procedure_name_list = ''
    for procedure in pipeline.named_steps:
        procedure_name_list += procedure + ';'
    csv_values.append(procedure_name_list)
    csv_values.append(str(seed))
    if feature_selection_function is not None:
        csv_values.append(feature_selection_function.__name__)
    else:
        csv_values.append("None")
    if prediction_function is not None:
        csv_values.append(prediction_function.__name__)
    else:
        csv_values.append("None")
    if feature_selection_function is not None:
        foo = feature_selection_function(data, n_features, seed)
        aux = data['x'][foo]
        data['x'] = aux
    tested_data, predictions,results = prediction_function(pipeline, data, seed)
    csv_values.append(str(n_features))
    conf_matrix = confusion_matrix(tested_data, predictions)
    class_report = classification_report(tested_data, predictions, output_dict=True)
    csv_values.append(region)
    if visual:
        plot_confusion_matrix(tested_data, predictions, [0, 1],
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues)
        plt.show()
    with open('results.csv', 'a') as result_csv:
        writer = csv.writer(result_csv)
        csv_values.extend([
            class_report['0']['f1-score'],
            class_report['0']['precision'],
            class_report['0']['recall'],
            class_report['1']['f1-score'],
            class_report['1']['precision'],
            class_report['1']['recall'],
            accuracy_score(tested_data, predictions)]
        )
        writer.writerow(csv_values)

    print('Done')
    return results


def test_pipeline_state(data, data_state, pipeline, seed, n_features=16, feature_selection_function=None, prediction_function=None,
                  visual=False, region='All'):
    csv_values = [str(datetime.datetime.now())]
    csv_values_state = [str(datetime.datetime.now())]
    procedure_name_list = ''
    for procedure in pipeline.named_steps:
        procedure_name_list += procedure + ';'
    csv_values.append(procedure_name_list)
    csv_values_state.append(procedure_name_list)
    csv_values.append(str(seed))
    csv_values_state.append(str(seed))
    if feature_selection_function is not None:
        csv_values.append(feature_selection_function.__name__)
        csv_values_state.append(feature_selection_function.__name__)
    else:
        csv_values.append("None")
        csv_values_state.append("None")
    if prediction_function is not None:
        csv_values.append(prediction_function.__name__)
        csv_values_state.append(prediction_function.__name__)
    else:
        csv_values.append("None")
        csv_values_state.append("None")

    if feature_selection_function is not None:
        foo = feature_selection_function(data, n_features, seed)
        aux = data['x'][foo]
        data['x'] = aux
        foo = feature_selection_function(data_state, n_features, seed)
        aux = data_state['x'][foo]
        data_state['x'] = aux
    csv_values.append(str(n_features))
    csv_values_state.append(str(n_features))
    csv_values.append(region)
    csv_values_state.append(region)

    tested_data, predictions, results = prediction_function(pipeline, data, data_state, seed)
    conf_matrix = confusion_matrix(tested_data, predictions)
    class_report = classification_report(tested_data, predictions, output_dict=True)
    if visual:
        plot_confusion_matrix(tested_data, predictions, [0, 1],
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues)
        plt.show()
    with open('results_global.csv', 'a') as result_csv:
        writer = csv.writer(result_csv)
        csv_values.extend([
            class_report['0']['f1-score'],
            class_report['0']['precision'],
            class_report['0']['recall'],
            class_report['1']['f1-score'],
            class_report['1']['precision'],
            class_report['1']['recall'],
            accuracy_score(tested_data, predictions)]
        )
        writer.writerow(csv_values)

    tested_data, predictions,results = kfold_cross_val_predictions(pipeline, data_state, seed)
    conf_matrix = confusion_matrix(tested_data, predictions)
    class_report = classification_report(tested_data, predictions, output_dict=True)
    if visual:
        plot_confusion_matrix(tested_data, predictions, [0, 1],
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues)
        plt.show()
    with open('results_specific.csv', 'a') as result_csv:
        writer = csv.writer(result_csv)
        csv_values_state.extend([
            class_report['0']['f1-score'],
            class_report['0']['precision'],
            class_report['0']['recall'],
            class_report['1']['f1-score'],
            class_report['1']['precision'],
            class_report['1']['recall'],
            accuracy_score(tested_data, predictions)]
        )
        writer.writerow(csv_values_state)
    
    print('Done')
    return results


def test_pipeline_hybrid(data, pipeline, seed, n_features=16, feature_selection_function=None, prediction_function=None,
                  visual=False, region='All'):
    csv_values = [str(datetime.datetime.now())]
    procedure_name_list = ''
    for procedure in pipeline.named_steps:
        procedure_name_list += procedure + ';'
    csv_values.append(procedure_name_list)
    csv_values.append(str(seed))
    if feature_selection_function is not None:
        csv_values.append(feature_selection_function.__name__)
    else:
        csv_values.append("None")
    if prediction_function is not None:
        csv_values.append(prediction_function.__name__)
    else:
        csv_values.append("None")
    if feature_selection_function is not None:
        for i in data:
            foo = feature_selection_function(i, n_features, seed)
            aux = i['x'][foo]
            i['x'] = aux
    tested_data = []
    predictions = []
    results = []
    for i in data:
        tested_data_i, predictions_i, results_i = prediction_function(pipeline, i, seed)
        tested_data.extend(tested_data_i)
        predictions.extend(predictions_i)
        results.extend(results_i)
    csv_values.append(str(n_features))
    conf_matrix = confusion_matrix(tested_data, predictions)
    class_report = classification_report(tested_data, predictions, output_dict=True)
    csv_values.append(region)
    if visual:
        plot_confusion_matrix(tested_data, predictions, [0, 1],
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues)
        plt.show()
    with open('results_hybrid.csv', 'a') as result_csv:
        writer = csv.writer(result_csv)
        csv_values.extend([
            class_report['0']['f1-score'],
            class_report['0']['precision'],
            class_report['0']['recall'],
            class_report['1']['f1-score'],
            class_report['1']['precision'],
            class_report['1']['recall'],
            accuracy_score(tested_data, predictions)]
        )
        writer.writerow(csv_values)

    print('Done')
    return results

