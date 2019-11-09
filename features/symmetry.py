# coding: utf-8

# симметричность огибающей относительно нуля
# отдельно строятся огибающие для отсчётов больше нуля и отсчётов меньше нуля
# затем вычисляется корреляция между этими огибающими

import numpy as np


def symmetry(samples):
    win = np.ones(500) / 500
    positive_env = np.convolve(np.clip(samples, 0, samples.max()), win, mode='same')
    negative_env = np.convolve(-np.clip(samples, samples.min(), 0), win, mode='same')
    return np.corrcoef(positive_env, negative_env)[0, 1]
