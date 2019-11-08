# coding: utf-8

# вычисление полного набора признаков для всех дикторов
# признаки сохраняются в файлы отдельно для обучающей и тестовой выборок

import numpy as np
from utils.files import files
from utils.make_data_set import make_data_set
from preprocess.normalize import normalize
from features.mfcc import mfcc
from features.lpc import lpc
from features.formants import formant_features


def features_callback(samples, sample_rate):
    samples = normalize(samples)

    mfcc_feat = mfcc(samples, sample_rate, 13)    # 26 признаков
    lpc_feat = lpc(samples, sample_rate, 160, 5)  # 10 признаков
    formant_feat = formant_features(samples, sample_rate)  # 13 признаков

    return np.hstack((mfcc_feat, lpc_feat, formant_feat))


if __name__ == '__main__':
    samples_path = 'c:/work/projects/samples_16kHz_train_test'
    train_set, test_set = make_data_set(files(samples_path), 100000, 3, features_callback)

    np.save('train_set.npy', train_set)
    np.save('test_set.npy', test_set)
