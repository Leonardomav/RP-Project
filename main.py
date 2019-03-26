import pandas as pd

weather_dataframe_raw = pd.read_csv('weatherAUS.csv')
n_weather_dataframe = weather_dataframe_raw.dropna(1, thresh=len(weather_dataframe_raw.index) * 0.2)
print(n_weather_dataframe)
n_weather_dataframe = weather_dataframe_raw.dropna(0, how='any')
print(n_weather_dataframe)