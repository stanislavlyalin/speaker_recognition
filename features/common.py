# coding: utf-8

import numpy as np

# вычисление спектрограммы сигнала
def spectrogram(samples, fft_size=2048, step_size=256):

    hamming = np.hamming(fft_size)

    m = (len(samples) - fft_size) // step_size + 1
    n = fft_size // 2 + 1

    sp = np.empty((m, n))

    for i in range(m):
        block = hamming * samples[i*step_size:i*step_size + fft_size]
        sp[i] = np.abs(np.fft.rfft(block))

    return sp


# перевод частоты из бин в герцы
def bin_to_hz(f, fft_size, sample_rate):
    return f * sample_rate / fft_size
