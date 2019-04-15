import csv
import datetime
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from sklearn.model_selection import KFold, cross_val_predict



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def test_pipeline(data, pipeline, seed, prediction_function, visual=False):
    test_name = str(datetime.datetime.now()) + ":"
    for procedure in pipeline.named_steps:
        test_name += procedure + ';'
    test_name = test_name[:-1] + "-" + str(seed)
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
