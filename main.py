from matplotlib import pyplot
import numpy
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler

filename = 'weatherAUS.csv'
names = ['preg', 'plas', 'press', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data_raw = read_csv(filename, names=names)

#print(data_raw)

# Remove features that have more than 20% of missing values
data = data_raw.dropna(1, thresh=len(data_raw.index) * 0.2)
#print(data)

# Remove examples that have any missing values
data = data.dropna(0, how='any')
print(data)

# peek = data.head(20)  # le as primeiras 20 linhas
# types = data.dtypes
# set_option('display.width', 100)
# set_option('precision', 3)
# description = data.describe()
# class_counts = data.groupby('class').size()
# correlation = data.corr(method='pearson')
# skew = data.skew()
# data.hist()
# data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
# data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
# scatter_matrix(data)

# correlation matrix
# fig = pyplot.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlation, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = numpy.arange(0, 9, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)

# rescale data
# array = data.values
# separate array into input and output components
# X = array[:, 0:8]
# Y = array[:, 8]
# scaler = MinMaxScaler(feature_range=(0,1))
# rescaledX = scaler.fit(X)
# summarize transformed data
# numpy.set_printoptions(precision=3)

# pyplot.show()
# print(peek)
# print(data.shape)
# print(types)
# print(class_counts)
# print(description)
# print(correlation)
# print(skew)
