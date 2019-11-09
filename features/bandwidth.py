# coding: utf-8

# средняя полоса сигнала или изменение полосы сигнала

# вычисляется огибающая сигнала и с помощью порога отсекаются низкоэнергетические участки в начале и конце сигнала
# по оставшейся части вычисляется спектрограмма в логарифмическом масштабе
# каждая линия спектрограммы сглаживается окном
# на сглаженной линии справа находится такая граница, которая превышает порог - это и будет полосой для данной линии
# все полосы усредняются, вычисляется их дисперсия

import numpy as np
from preprocess.normalize import normalize
from .common import spectrogram, bin_to_hz


def bandwidth(samples, sample_rate):

    # нормализация отсчётов для того, чтобы потом воспользоваться порогом
    samples = normalize(samples)

    # вычисление огибающей сигнала
    win_size = int(50 * sample_rate / 1000)  # 30 миллисекунд
    conv = np.convolve(np.abs(samples), np.ones(win_size) / win_size, mode='same')

    # поиск границ сигнала
    threshold = 0.2
    start, end = 0, len(conv) - 1

    while conv[start] < threshold:
        start += 1

    while conv[end] < threshold:
        end -= 1

    # получение спектрограммы сигнала
    sp = 10 * np.log10(spectrogram(samples[start:end]))

    smooth_size = 100
    smooth_win = np.ones(smooth_size) / smooth_size
    threshold = 6.0

    bandwidths = []

    for i, line in enumerate(sp):
        conv = np.convolve(line, smooth_win, mode='same')

        end = len(conv) - 1
        while conv[end] < threshold and end > 0:
            end -= 1

        line_band = bin_to_hz(end, sp.shape[1], sample_rate)
        bandwidths.append(line_band)

    return np.mean(bandwidths), np.var(bandwidths)
