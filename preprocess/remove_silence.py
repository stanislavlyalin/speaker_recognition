import numpy as np

def remove_silence(samples, win_size, threshold):
    win = np.ones(win_size) / win_size
    smooth = np.convolve(np.abs(samples), win, mode='same')
    return samples[smooth > threshold]
