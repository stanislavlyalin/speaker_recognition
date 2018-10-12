import numpy as np

def samples_to_data_sets(samples, data_set_len, n):
    data_sets = []

    for i in range(0, len(samples) - data_set_len, data_set_len):
        data_set = np.empty((0, n+1))
        block = samples[i:i+data_set_len]

        for j in range(data_set_len - n):
            data_set = np.vstack((data_set, block[j:j+n+1]))
        data_sets.append(data_set)

    return np.array(data_sets)
