import scipy
import numpy as np

def kruskal_wallis(data):
    rank=[]
    for i in range(data.shape[1]-3):
        setX = data[data.columns[i+2]].tolist()
        setY = data["RainTomorrow"].tolist()
        rank.append(scipy.stats.kruskal(setX,setY))
    print(rank)