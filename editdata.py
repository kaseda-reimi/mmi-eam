import numpy as np
import function as fc

max_op = 0.9
min_op = 0
max_ex = 30
min_ex = 0 #０以下を評価で除外とか
max_w2 = 0.8
min_w2 = 0.1
max_l1 = 0.7
min_l1 = 0
max_l2 = 0.9
min_l2 = 0

def main():
    data_r = fc.get_data_r()
    w1,L,w2,l1,l2, _,_,output_off,_, _,_,output_on,_= data_r.T

    percent_w2 = (w1 - w2) / w1 /2
    percent_l2 = (L - l1 - l2) / L
    percent_l1 = l1 / L
    extinction_ratio = 20 * np.log10 (output_off / output_on)#消光比

    normalized_w1 = normalize_w1(w1)
    normalized_w2 = normalize_w2(percent_w2)
    normalized_l1 = normalize_l1(percent_l1)
    normalized_l2 = normalize_l2(percent_l2)
    normalized_ex = normalize_extinction_ratio(extinction_ratio)
    normalized_op = normalize_output_off(output_off)
    
    data = np.array([normalized_w1, normalized_w2, normalized_l1, normalized_l2, normalized_op, normalized_ex]).T

    path = 'mmi-eam/data.txt'
    write_data(path, data)



def normalize_w1(w1):
    max_w1 = 4.5e-6
    min_w1 = 1e-6
    normalized_w1 = (w1 - min_w1) / (max_w1-min_w1)
    return normalized_w1

def normalize_w2(w2):
    normalized_w2 = (w2 - min_w2) / (max_w2-min_w2)
    return normalized_w2

def normalize_l1(l1):
    normalized_l1 = (l1 - min_l1) / (max_l1 - min_l1)
    return normalized_l1

def normalize_l2(l2):
    normalized_l2 = (l2 - min_l2) / (max_l2 - min_l2)
    return normalized_l2

def normalize_extinction_ratio(extinction_ratio):
    normalized_ex = (extinction_ratio - min_ex) / (max_ex - min_ex)
    return normalized_ex

def normalize_output_off(output_off):
    normalized_op = (output_off - min_op) / (max_op - min_op)
    return normalized_op

def write_data(path, data):
    with open(path, mode='w') as f:
        for i in range(data.shape[0]):
            for n in data[i]:
                f.write(str(n))
            f.write('\n')

def denormalize_w2(normalized_w2):
    denormalized_w2 = normalized_w2 * (max_w2 - min_w2) + min_w2
    return denormalized_w2

def denormalize_l1(normalized_l1):
    denormalized_l1 = normalized_l1 * (max_l1 - min_l1) + min_l1
    return denormalized_l1

def denormalize_l2(normalized_l2):
    denormalized_l2 = normalized_l2 * (max_l2 - min_l2) + min_l2
    return denormalized_l2

def denormalize_extinction_ratio(normalized_ex):
    denormalized_ex = normalized_ex * (max_ex - min_ex) + min_ex
    return denormalized_ex

def denormalize_output_off(normalized_op):
    denormalized_op = normalized_op * (max_op - min_op) + min_op
    return denormalized_op


if __name__ == '__main__':
    main()
