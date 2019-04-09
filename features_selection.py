import scipy
import numpy as np

def kruskal_wallis(data):
    rank=[]
    for i in range(data.shape[1]):
        setX = data[data.columns[i]].tolist()
        setY = data["RainTomorrow"].tolist()
        rank.append((i,scipy.stats.kruskal(setX,setY)))
    rank = sorted(rank, key=lambda x: x[1]  )
    print(rank)