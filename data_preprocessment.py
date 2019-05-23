import matplotlib
import pandas

matplotlib.use('TkAgg')


def categorize_data(data):
    """
    Replaces all cardinal points with an int value

    Recieves:
        data -> dataframe
    Returns:
        data -> categorized dataframe
    """

    labels = data['WindGustDir'].astype('category').cat.categories.tolist()
    col = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

    for c in col:
        replace_map = {c: {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
        data.replace(replace_map, inplace=True)

    return data


def get_preprocessed_data():
    """
    Main preprocessment function
    Loads file and removes unnecessary columns and rows from the data frame and calls categorization function

    Returns:
        {'x': data, 'y': data_y} -> new data structure 
        states -> list of available states
        len(data.columns) -> num of features
        data_loc -> data structure with feature location
    
    """

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

    # Categorize
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})

    data = categorize_data(data)

    data_y = data['RainTomorrow'].ravel()
    data_loc = data.drop(['Date'], axis=1)
    data = data_loc.drop(['Location', 'RainTomorrow'], axis=1)

    return {'x': data, 'y': data_y}, states, len(data.columns), data_loc


def select_location(data_loc, loc):
    """
    This function returns a new data frame with only measurements from the selected location
    Recieves:
        data_loc -> data structure with feature location
        loc -> location selected
    Returns:
        {'x': data, 'y': data_y} -> new data structure 
    """
    data_loc = data_loc.loc[data_loc['Location'] == loc]
    data_y = data_loc['RainTomorrow'].ravel()
    data_loc = data_loc.drop(['Location', 'RainTomorrow'], axis=1)

    return {'x': data_loc, 'y': data_y}
