import matplotlib
from matplotlib import pyplot

matplotlib.use('TkAgg')

import pandas
import sklearn.preprocessing
import sklearn.model_selection
import draft.data_info
from draft import feature_selection
import feature_selection_pipeline


# This file is used as a notebook to explore the data

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


def normalize_data(data):
    # normaliza data
    data_normalized = data.drop(['Date', 'Location'], axis=1)
    x = data_normalized.values
    scaler = sklearn.preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    data_normalized = pandas.DataFrame(x_scaled, columns=data_normalized.columns)

    data_y_norm = data_normalized[['RainTomorrow']]

    data_normalized = data_normalized.drop(['RainTomorrow'], axis=1)

    return data_normalized, data_y_norm, x_scaled


# DEPRECATED
def get_data_kkw(n_features, data, KKW_rank):
    keep_index = []
    for i in range(n_features):
        keep_index.append(KKW_rank[i][0])

    data_kkw = data.iloc[:, keep_index]

    return data_kkw


def data_split(data, data_y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data_y, test_size=0.25,
                                                                                random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test


def main():
    # Load data set
    filename = 'weatherAUS.csv'
    data_raw = pandas.read_csv('weatherAUS.csv')

    # Remove features that have more than 20% of missing values
    data_less_raw = data_raw.dropna(1, thresh=len(data_raw.index) * 0.8)

    # Remove examples that have any missing values
    data_less_raw = data_less_raw.dropna(0, how='any')

    # Remove RISK_MM
    data_less_raw = data_less_raw.drop(['RISK_MM'], axis=1)

    # Print general info about the data
    #draft.data_info.print_general_information(data_less_raw)

    # Describe a Single Columns
    #draft.data_info.describe_all_columns(data_less_raw)

    # Print correlation between all features or between two if specified
    #draft.data_info.correlation_info(data_less_raw, 'pearson')

    # Print data skew
    # print(data_less_raw.skew())

    # Plot the features histogram
    #data_less_raw.hist()
    #matplotlib.pyplot.show()

    # Kruskal_wallis

    # change Yes and No to 1 and 0
    #
    #
    states = ["All"]
    states.extend(data_less_raw['Location'].unique())

    data = data_less_raw.copy()

    # Categorize
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})

    data = categorize_data(data)

    data_y = data['RainTomorrow'].ravel()
    data_loc = data.drop(['Date'], axis=1)

    data, _, _ = normalize_data(data)

    return {'x': data, 'y': data_y}, states, len(data.columns), data_loc


if __name__ == '__main__':
    main()
