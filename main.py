from matplotlib import pyplot
import numpy
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
import data_info

# Load data set
filename = 'weatherAUS.csv'
data_raw = read_csv(filename)

# Remove features that have more than 20% of missing values
data = data_raw.dropna(1, thresh=len(data_raw.index) * 0.8)

# Remove examples that have any missing values
data = data.dropna(0, how='any')

# Print general info about the data
# data_info.print_general_information(data)

# Describe a Single Columns
# data_info.describe_column(data, 'Location')

# Print correlation between all features or between two if specified
# data_info.correlation_info(data, 'pearson', "MinTemp", "MaxTemp")

# Print data swew
# print(data.skew())

# Plot the features histogram
# data.hist()
# pyplot.show()
