# coding: utf-8

# аппроксимация среднего спектра полиномом заданной степени
# используется спектр в логарифмической шкале
# для вычисления коэффициентов полинома используется LinearRegression из sklearn

from preprocess.normalize import normalize
import numpy as np
from .common import spectrogram
from sklearn.linear_model import LinearRegression


def polynomial_coefs(samples):
    samples = normalize(samples)

    sp = 10 * np.log10(spectrogram(samples))
    mean_sp = sp.mean(axis=0)[:512]

    max_power = 5
    col1 = np.arange(len(mean_sp)).reshape((-1, 1))
    X = np.ones(col1.shape)
    y = mean_sp.reshape(X.shape)

    for i in range(max_power):
        X = np.c_[X, col1 ** (i + 1)]

    model = LinearRegression().fit(X, y)
    coefs = [model.intercept_[0]] + list(model.coef_[0][1:])

    return coefs
