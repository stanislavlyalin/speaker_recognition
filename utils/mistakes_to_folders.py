# coding: utf-8

# Скрипт предназначен для раскладывания по каталогам файлов, на которых классификатор ошибся.

# алгоритм работы:
# 1. найти те классы (id пользователей), на которых классификатор ошибся,
#    и составить массив пар: ошибочный id - верный id
# 2. формирование рабочих каталогов с файлами для анализа.
#    На каждый из ошибочных примеров будет по 8 файлов: 4 файла с нужным классом
#    и 4 файла с ошибочным классом.

import os
import shutil
import re

def mistakes_to_folders(model, X_test, y_test, wav_files, mistakes_dir):

    # создание массива пар ошибочных предсказаний: актуальное значение vs предсказанное
    predicted = model.predict(X_test)
    mistake_actual = y_test[predicted != y_test]
    mistake_predicted = predicted[predicted != y_test]

    # for i in range(len(mistake_actual)):
    #     print('Actual %d, predicted %d' % (mistake_actual[i], mistake_predicted[i]))
    # print('Total %d mistakes' % len(mistake_actual))

    # формирование рабочих каталогов с файлами для анализа
    if os.path.exists(mistakes_dir):
        shutil.rmtree(mistakes_dir, ignore_errors=True)
        os.makedirs(mistakes_dir)

    for i in range(len(mistake_actual)):
        correct_id = mistake_actual[i]
        incorrect_id = mistake_predicted[i]

        # print('actual class %d, predicted class %d' % (correct_id, incorrect_id))

        path = '%s%04d_actual_%d_predicted_%d/' % (mistakes_dir, i, correct_id, incorrect_id)
        os.makedirs(path)

        regex = re.compile(r'%04d' % correct_id)
        correct_files = list(filter(regex.search, wav_files))

        regex = re.compile(r'%04d' % incorrect_id)
        incorrect_files = list(filter(regex.search, wav_files))
        
        for f in range(len(correct_files)):
            shutil.copy(correct_files[f], path)
            shutil.copy(incorrect_files[f], path)
