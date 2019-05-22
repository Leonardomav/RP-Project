from tkinter import *
from tkinter.ttk import *
import matplotlib
from test_pipeline import test_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import PCA
from feature_selection_pipeline import kruskal_wallis, select_k_best, ROC
from prediction_pipeline import kfold_cross_val_predictions, train_test_predictions
import data_preprocessment as dp

matplotlib.use('TkAgg')


def GUI(data, data_loc, states, n_features, feature_sel, dim_reduction, predict_methods, classifiers):
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

    def clicked(data):
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
            if comb0.get() != "All":
                data = dp.select_location(data_loc, comb0.get())

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
                    n_features=int(comb2.get()),
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
                    n_features=int(comb2.get()),
                    feature_selection_function=feature_selection_function,
                    prediction_function=prediction_function)
                warText.configure(text="")
        except Exception as e:
            print(e)
            warText.configure(text="This combination is invalid")

    btn = Button(window, text="Run Configuration", command=lambda: clicked(data))

    btn.grid(column=1, row=7, columnspan=2)

    window.geometry('655x275')
    window.mainloop()


def main():
    data, states, num_columns, data_loc = dp.get_preprocessed_data()

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

    GUI(data, data_loc, states, num_columns, feature_selection_functions, fit_transform_options, prediction_functions,
        classifiers)


if __name__ == '__main__':
    main()
