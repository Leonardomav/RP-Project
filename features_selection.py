import scipy

def kruskal_wallis(data):
    rank=[]
    setX = data.loc[:, data.columns != "RainTomorrow"]
    setY = data["RainTomorrow"].tolist()
    scipy.stats.kruskal(setX,setY)