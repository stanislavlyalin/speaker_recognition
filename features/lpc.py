import numpy as np
from utils.samples_to_data_sets import samples_to_data_sets

def lpc(samples, sample_rate, block_len, count):
    lpc_coefs = np.empty((0, count))

    for data_set in samples_to_data_sets(samples, block_len, count-1):
        ones = np.ones((data_set.shape[0], 1))
        X = np.hstack((ones, data_set[:,:-1]))
        y = data_set[:,-1]

        theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
        lpc_coefs = np.vstack((lpc_coefs, theta.reshape((1, -1))))

    return np.hstack((np.mean(lpc_coefs, axis=0), np.var(lpc_coefs, axis=0)))
