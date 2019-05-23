import matplotlib
import pandas
import sklearn

matplotlib.use('TkAgg')


def categorize_data(data):
    labels = data['WindGustDir'].astype('category').cat.categories.tolist()
    col = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

    for c in col:
        replace_map = {c: {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
        data.replace(replace_map, inplace=True)

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
    data_loc = data.drop(['Date'], axis=1)

    data, _, _= normalize_data(data)

    return {'x': data, 'y': data_y}, states, len(data.columns), data_loc


def select_location(data_loc, loc):
    data_loc = data_loc.loc[data_loc['Location'] == loc]
    data_y = data_loc['RainTomorrow'].ravel()
    data_loc = data_loc.drop(['Location', 'RainTomorrow'], axis=1)

    return {'x': data_loc, 'y': data_y}
