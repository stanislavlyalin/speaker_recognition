# coding: utf-8

# Скрипт предназначен для раскладывания по каталогам файлов, на которых классификатор ошибся.
# В качестве классификатора используется SVC с C=10. Среди всех признаков, представленных в
# выборке, берутся только те, что перечислены в indexes, на которых модель показана наибольшую точность: 90%.

# алгоритм работы:
# 1. создать и обучить классификатор SVC с параметрами, дающими
#    точность 90% на выборке из 1220 пользователей
# 2. найти те классы (id пользователей), на которых классификатор ошибся,
#    и составить массив пар: ошибочный id - верный id
# 3. формирование рабочих каталогов с файлами для анализа.
#    На каждый из ошибочных примеров будет по 8 файлов: 4 файла с нужным классом
#    и 4 файла с ошибочным классом.

# TODO: реализовать в виде функции, которая принимает выборки, модель и каталог с файлами, соответствующими выборке, а также каталог, в котором нужно разложить файлы

import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle
import shutil
import glob
import re

# 1. создание и обучение классификатора
# подготока обучающей и тестовой выборок
train_set = np.load('train_set.npy')
test_set = np.load('test_set.npy')

X_train = np.nan_to_num(train_set[:, :-1])
X_test = np.nan_to_num(test_set[:, :-1])

y_train = train_set[:, -1].astype(int)
y_test = test_set[:, -1].astype(int)

# стандартизация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# создание и обучение классификатора
from sklearn.svm import SVC

indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 27, 28]

model_filename = 'model_svc.dump'

if os.path.exists(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
else:
    model = SVC(C=10)
    model.fit(X_train[:, indexes], y_train)
    
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

# 2. создание массива пар ошибочных предсказаний: актуальное значение vs предсказанное
predicted = model.predict(X_test[:, indexes])
mistake_actual = y_test[predicted != y_test]
mistake_predicted = predicted[predicted != y_test]

for i in range(len(mistake_actual)):
    print('Actual %d, predicted %d' % (mistake_actual[i], mistake_predicted[i]))
print('Total %d mistakes' % len(mistake_actual))

# 3. формирование рабочих каталогов с файлами для анализа
mistakes_dir = './mistakes_dir/'

if os.path.exists(mistakes_dir):
    shutil.rmtree(mistakes_dir, ignore_errors=True)
    os.makedirs(mistakes_dir)

wav_files = glob.glob('c:/work/projects/samples_16kHz_train_test/*.wav')

for i in range(len(mistake_actual)):
    correct_id = mistake_actual[i]
    incorrect_id = mistake_predicted[i]

    print('actual class %d, predicted class %d' % (correct_id, incorrect_id))

    path = '%s%04d_actual_%d_predicted_%d/' % (mistakes_dir, i, correct_id, incorrect_id)
    os.makedirs(path)

    regex = re.compile(r'%04d' % correct_id)
    correct_files = list(filter(regex.search, wav_files))

    regex = re.compile(r'%04d' % incorrect_id)
    incorrect_files = list(filter(regex.search, wav_files))
    
    for f in range(len(correct_files)):
        shutil.copy(correct_files[f], path)
        shutil.copy(incorrect_files[f], path)
