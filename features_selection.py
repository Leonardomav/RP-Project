import scipy

def kruskal_wallis(data, data_y):
    rank=[]
    setY = data_y['RainTomorrow'].tolist()
    for i in range(data.shape[1]):
        setX = data[data.columns[i]].tolist()
        rank.append((i,scipy.stats.kruskal(setX,setY)[0]))



    rank = sorted(rank, key=lambda x: x[1])
    return rank
