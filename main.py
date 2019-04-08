from matplotlib import pyplot
import numpy
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
import scipy
import data_info
import features_selection

# Load data set
filename = 'weatherAUS.csv'
data_raw = read_csv(filename)

# Remove features that have more than 20% of missing values
data = data_raw.dropna(1, thresh=len(data_raw.index) * 0.8)

# Remove examples that have any missing values
data = data.dropna(0, how='any')

#change Yes and No to 1 and 0
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0}).astype('category')
data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0}).astype('category')


labels = data['WindGustDir'].astype('category').cat.categories.tolist()
col = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

for c in col:
    replace_map = {c : {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
    data.replace(replace_map, inplace=True)
    data[c].astype('category')

labels = data['Location'].astype('category').cat.categories.tolist()
replace_map = {'Location': {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
data.replace(replace_map, inplace=True)
data['Location'].astype('category')


# Print general info about the data
#data_info.print_general_information(data)

# Describe a Single Columns
# data_info.describe_column(data)

# Print correlation between all features or between two if specified
#data_info.correlation_info(data, 'pearson')

# Print data skew
# print(data.skew())

# Plot the features histogram
# data.hist()
# pyplot.show()


#Kruskal_wallis
features_selection.kruskal_wallis(data)





