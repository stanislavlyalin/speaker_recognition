import numpy as np


# из герц в шкалу мел
def mel(f):
    return 1127 * np.log(1 + f / 700)


# из мел в герцы
def freq(m):
    return 700 * (np.exp(m / 1127) - 1)


# из герц в бины спектра
def bin(f, sampleRate, fftSize):
    return int(f * fftSize / sampleRate)


# генерация треугольных фильтров
def generateFilters(minF, maxF, count, sampleRate, fftSize):

    melPoints = np.linspace(mel(minF), mel(maxF), count + 2)
    points = [bin(freq(m), sampleRate, fftSize) for m in melPoints]

    filters = np.zeros((count, fftSize // 2))

    for i in range(1, count + 1):
        for k in range(filters.shape[1]):
            value = 0

            if points[i - 1] <= k <= points[i]:
                value = (k - points[i - 1]) / (points[i] - points[i - 1])
            elif points[i] <= k <= points[i + 1]:
                value = (points[i + 1] - k) / (points[i + 1] - points[i])

            filters[i-1, k] = value
    return filters


# дискретное косинусное преобразование-2 с нормировкой
def dctNorm(x):

    N = len(x)
    y = np.zeros(N)

    for k in range(N):

        items = [x[n] * np.cos(np.pi * k * (n + 0.5) / N) for n in range(N)]
        value = 2 * np.sum(items)
        value *= np.sqrt(1 / (4 * N)) if k == 0 else np.sqrt(1 / (2 * N))

        y[k] = value

    return y


def sampleCount(duration, sampleRate):
    return int(duration * sampleRate)


def signalWindowSize(duration, sampleRate):
    return sampleCount(duration, sampleRate)


def spectrumWindowSize(duration, sampleRate):
    return signalWindowSize(duration, sampleRate) // 2


def mfcc(x, sampleRate, blockDuration=0.025, blockOverlap=0.010,
    minFrequency=300, maxFrequency=8000, mfccCount=13,
    signalWindow=None, spectrumWindow=None, squareSpectrum=False):
    
    fftSize = sampleCount(blockDuration, sampleRate)
    stepSize = fftSize - sampleCount(blockOverlap, sampleRate)

    filters = generateFilters(minFrequency, maxFrequency,
        mfccCount, sampleRate, fftSize)

    _mfcc = np.array([])

    for i in range(0, len(x) - fftSize, stepSize):

        block = x[i:i+fftSize]
        
        if signalWindow:
            block *= signalWindow
        
        sp = (np.square(np.absolute(
            np.fft.rfft(block, n=fftSize))) / fftSize)[:fftSize//2]

        if spectrumWindow:
            sp *= spectrumWindow

        if squareSpectrum:
            sp = np.square(sp)
        
        S = [np.log(np.sum(sp * filters[j])) for j in range(mfccCount)]

        _mfcc = np.concatenate((_mfcc, np.array(dctNorm(S))))

    return _mfcc.reshape((-1, mfccCount))
