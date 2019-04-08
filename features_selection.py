import scipy
import numpy as np

def kruskal_wallis(data):
    rank=[]
    rank = np.array(rank, dtype=np.float64)
    for i in range(data.shape[1]-3):
        setX = data[data.columns[i+2]].tolist()
        setY = data["RainTomorrow"].tolist()
        rank.append(rank, scipy.stats.kruskal(setX,setY))
    print(rank)