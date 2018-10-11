import numpy as np

def normalize(samples):
    return (samples - samples.mean()) / samples.std()
