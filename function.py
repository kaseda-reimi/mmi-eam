import numpy as np
import os

def get_data_r():
    path = os.getcwd()+'/data_r.txt'
    with open (path) as f:
        l = f.read().split()
    l = [float(s) for s in l]
    l = np.array(l).reshape(-1, 13)
    return l

def get_data():
    path = os.getcwd()+'/data.txt'
    with open (path) as f:
        l = f.read().split()    
    l = [float(s) for s in l]
    l = np.array(l).reshape(-1, 6)
    return l

def write_data(path, data):
    with open(os.getcwd()+path, mode='w') as f:
        for i in range(data.shape[0]):
            for n in data[i]:
                f.write(str(n))
            f.write('\n')