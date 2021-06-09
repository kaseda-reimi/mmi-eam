import numpy as np

def get_data_r():
    path = 'mmi-eam/data_r.txt'
    with open (path) as f:
        l = f.read().split()
    l = [float(s) for s in l]
    l = np.array(l).reshape(-1, 13)
    return l

def get_data():
    path = 'mmi-eam/data.txt'
    with open (path) as f:
        l = f.read().split()    
    l = [float(s) for s in l]
    l = np.array(l).reshape(-1, 6)
    return l
