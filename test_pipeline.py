import csv
import datetime
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from visualizations import plot_confusion_matrix


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
