# coding: utf-8

# отношение средней длительности слов к средней длительности пауз

import numpy as np


def signal_pause_ratio(samples, sample_rate):

    win_size = int(50 * sample_rate / 1000)  # 50 миллисекунд
    conv = np.convolve(np.abs(samples), np.ones(win_size) / win_size, mode='same')

    threshold = 0.2

    start = 0
    while conv[start] < threshold:
        start += 1

    end = len(conv) - 1
    while conv[end] < threshold:
        end -= 1
    
    if start < end and start > 0:

        positive = 0
        negative = 0

        positives = []
        negatives = []

        for i in range(start, end + 1):
            if conv[i] >= threshold:
                if conv[i-1] < threshold:
                    positive = 1
                    if negative > 1:
                        negatives.append(negative)
                else:
                    positive += 1
            else:
                if conv[i-1] >= threshold:
                    negative = 1
                    if positive > 1:
                        positives.append(positive)
                else:
                    negative += 1
        
        if len(positives) > 0 and len(negatives) > 0:
            return np.mean(positives) / np.mean(negatives)
    
    return 0
