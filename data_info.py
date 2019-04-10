from pandas import set_option, option_context

def print_general_information(data):
    types = data.dtypes
    class_counts = data.groupby('RainTomorrow').size()

    print("Data Shape and all the features:")
    print(data.shape)
    print(types)
    print("\nClass Counts:")
    print(class_counts)


def describe_column(data, column):
    set_option('display.width', 1000)
    set_option('precision', 3)
    print('\nDescribing ' + column + ':')
    print(data[column].describe())


def describe_all_columns(data):
    set_option('display.width', 1000)
    set_option('precision', 3)
    print('\nDescribing all columns:')
    print(data.describe())


def correlation_info(data, method, c1=None, c2=None):
    if c1 is None or c2 is None:
        print("\nCorrelation Table:")
        with option_context('display.max_rows', None, 'display.max_columns', None):
            print(data.corr(method=method))
    else:
        correlation = data[c1].corr(data[c2], method='pearson')
        print("\nCorrelation between " + c1 + " and " + c2 + ": ")
        print(correlation)
