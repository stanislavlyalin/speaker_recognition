from python_speech_features import mfcc as psf_mfcc
import numpy as np

def mfcc(samples, sample_rate, count):
    mfcc_feat = psf_mfcc(samples, sample_rate, numcep=count)
    return np.hstack((np.mean(mfcc_feat, axis=0), np.var(mfcc_feat, axis=0)))
