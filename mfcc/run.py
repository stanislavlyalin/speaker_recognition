from scipy.io.wavfile import read
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import numpy as np
from mfcc_features import mfcc as _mfcc

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# читаем файл
sampleRate, x = read('usr0001_male_youth_008.wav')

fftSize=1024
mfccCount = 10
minFrequency=300
maxFrequency=8000

print('Own implementation')
my_mfcc = _mfcc(x, sampleRate, blockDuration=fftSize/sampleRate, blockOverlap=0,
    minFrequency=minFrequency, maxFrequency=maxFrequency, mfccCount=mfccCount)
print(my_mfcc)

print('MFCC from Library')
mfcc_feat = mfcc(x, sampleRate, winlen=fftSize/sampleRate,\
    winstep=fftSize/sampleRate, numcep=mfccCount, nfilt=mfccCount,\
    nfft=fftSize, lowfreq=minFrequency, highfreq=maxFrequency,\
    preemph=0, ceplifter=0, appendEnergy=False)
print(mfcc_feat)

# for i in range(5):
#     plt.plot(my_mfcc[i], 'b--')
#     plt.plot(mfcc_feat[i], 'r--')
# plt.grid()
# plt.show()