import os
import glob
import re
import numpy as np
from features import features

samples_dir = 'c:/work/projects/GenderRecognition/samples'
os.chdir(samples_dir)

files = glob.glob('*.wav')


# извлечение идентификатора из имени файла
def id_from_filename(file_name):
    return int(re.search('\d{4}', file_name)[0])


X = np.array([])
y = np.array([], dtype=int)

for file in files:
    id = id_from_filename(file)

    if len(np.where(y == id)[0]) == 4:
        continue

    y = np.r_[y, id]
    X = np.r_[X, features(file)]

X = X.reshape((-1, 10))

print(X[:10,:])
print(y[:10])
