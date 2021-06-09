import numpy as np

def get_data_r():
    path = '/Users/kasedareibi/アーカイブ/Desktop/neural/data_r.txt'
    with open (path) as f:
        l = f.read().split()
    l = [float(s) for s in l]
    l = np.array(l).reshape(-1, 13)
    return l

def get_data():
    path = '/Users/kasedareibi/アーカイブ/Desktop/neural/data.txt'
    with open (path) as f:
        l = f.read().split()    
    l = [float(s) for s in l]
    l = np.array(l).reshape(-1, 6)
    return l
