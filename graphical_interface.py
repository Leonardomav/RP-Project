from tkinter import *
from tkinter.ttk import *
import matplotlib

matplotlib.use('TkAgg')
from test_pipeline import test_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
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
    states = ["All"]
    states.extend(data_less_raw['Location'].unique())

    data = data_less_raw.copy()
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})

    data = categorize_data(data)

    data_y = data['RainTomorrow'].ravel()
    data = data.drop(['Date', 'Location', 'RainTomorrow'], axis=1)

    return {'x': data, 'y': data_y}, states, len(data.columns)


def GUI(data, states, n_features, feature_sel, dim_reduction, predict_methods, classifiers):
    window = Tk()
    window.title("RP - GUI")
    title = Label(window, text='Weather In Australia - GUI', font=("Arial Bold", 20))
    title.grid(row=0, columnspan=4, pady=(5, 10))

    lbl1 = Label(window, text='Australian State:', font=("Arial", 12))
    lbl1.grid(column=0, row=1, pady=5, padx=(10, 0), sticky="W")

    comb0 = Combobox(window, state="readonly")
    comb0['values'] = tuple(states)
    comb0.current(0)  # set the selected item
    comb0.grid(column=1, row=1, padx=20)

    lbl2 = Label(window, text='Feature Selection:', font=("Arial", 12))
    lbl2.grid(column=0, row=2, pady=5, padx=(10, 0), sticky="W")

    comb1 = Combobox(window, state="readonly")
    iter_feature = iter(feature_sel)
    next(iter_feature)
    feat = [None]
    feat.extend([i.__name__ for i in iter_feature])
    comb1['values'] = feat
    comb1.current(0)  # set the selected item
    comb1.grid(column=1, row=2, padx=20)

    lbl3 = Label(window, text='N of Features:', font=("Arial", 12))
    lbl3.grid(column=2, row=2, pady=5, padx=(0, 15), sticky="W")

    comb2 = Combobox(window, state="readonly")
    comb2['values'] = [i + 1 for i in range(n_features)]
    comb2.current(0)  # set the selected item
    comb2.grid(column=3, row=2)

    lbl3 = Label(window, text='Dimensionality Reduction:', font=("Arial", 12))
    lbl3.grid(column=0, row=3, pady=5, padx=(10, 0), sticky="W")

    comb3 = Combobox(window, state="readonly")
    iter_dim = iter(dim_reduction)
    next(iter_dim)
    dim = [None]
    dim.extend([i[0] for i in iter_dim])
    comb3['values'] = dim
    comb3.current(0)  # set the selected item
    comb3.grid(column=1, row=3, padx=20)

    lbl3 = Label(window, text='N of Features:', font=("Arial", 12))
    lbl3.grid(column=2, row=3, pady=5, padx=(0, 15), sticky="W")

    comb4 = Combobox(window, state="readonly")
    comb4['values'] = [i + 1 for i in range(n_features)]
    comb4.current(0)  # set the selected item
    comb4.grid(column=3, row=3)

    lbl4 = Label(window, text='Prediction Method:', font=("Arial", 12))
    lbl4.grid(column=0, row=4, pady=5, padx=(10, 0), sticky="W")

    comb5 = Combobox(window, state="readonly")
    comb5['values'] = [i.__name__ for i in predict_methods]
    comb5.current(0)  # set the selected item
    comb5.grid(column=1, row=4, padx=20)

    lbl5 = Label(window, text='Classifier:', font=("Arial", 12))
    lbl5.grid(column=0, row=5, pady=5, padx=(10, 0), sticky="W")

    comb6 = Combobox(window, state="readonly")
    comb6['values'] = [i[0] for i in classifiers]
    comb6.current(0)  # set the selected item
    comb6.grid(column=1, row=5, padx=20)

    warText = Label(window, text='', font=("Arial", 8), foreground="red")
    warText.grid(column=1, row=6, columnspan=2)

    def clicked():
        seed = 3030

        for i in feature_sel:
            if comb1.get() == "None":
                feature_selection_function = None
                break

            elif i is not None and comb1.get() == i.__name__:
                feature_selection_function = i
                break

        for i in predict_methods:
            if comb5.get() == i.__name__:
                prediction_function = i

        for i in dim_reduction:
            if comb3.get() == "None":
                fit_transform_option = None
                break
            elif i is not None and comb3.get() == i[0]:
                fit_transform_option = i
                break

        for i in classifiers:
            if comb6.get() == "None":
                classifier = None
                break
            elif comb6.get() == i[0]:
                classifier = i
                break

        try:
            print("Run config")
            if comb6.get() == 'mahalanobis' and comb1.get() != "None" and comb3.get() == 'lda-dr':
                warText.configure(text="This combination is invalid")

            elif comb3.get() != "None":
                test_pipeline(
                    data,
                    Pipeline([
                        fit_transform_option,
                        classifier
                    ]),
                    seed,
                    feature_selection_function=feature_selection_function,
                    prediction_function=prediction_function)
                warText.configure(text="")

            else:
                test_pipeline(
                    data,
                    Pipeline([
                        classifier
                    ]),
                    seed,
                    feature_selection_function=feature_selection_function,
                    prediction_function=prediction_function)
                warText.configure(text="")
        except Exception as e:
             print(e)
             warText.configure(text="This combination is invalid")

    btn = Button(window, text="Run Configuration", command=clicked)

    btn.grid(column=1, row=7, columnspan=2)

    window.geometry('655x275')
    window.mainloop()


data, states, num_columns = get_preprocessed_data()

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

GUI(data, states, num_columns, feature_selection_functions, fit_transform_options, prediction_functions,
    classifiers)
