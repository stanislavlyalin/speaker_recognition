# coding: utf-8

# стабильность частоты основного тона

import numpy as np


def pitch_periodicity(samples, sample_rate):

    block_time = 0.050  # 50 milliseconds
    block_size = int(block_time * sample_rate)
    pitch_min, pitch_max = 50, 250  # min pitch 50 Hz, max - 250 Hz

    # под периодичностью ЧОТ (частоты основного тона) понимается максимальная средняя корреляция кусочков каждого блока
    # исходный сигнал разбивается на блоки
    # каждый блок разбивается на кусочки размера от 50 до 250 Гц (от минимальной ЧОТ до максимальной)
    # между всеми парами кусочков вычисляем коэффициент корреляции Пирсона
    # затем выбираем такое разбиение, при котором средняя корреляция оказывается максимальной
    # размер кусочка в данном случае будет равняться ЧОТ, а максимальная корреляция, равная 1, будет говорить о том, что сигнал идеально повторяется внутри блока

    periodicity = []  # список максимальных средних корреляций

    for i in range(0, len(samples), block_size):
        block = samples[i:i + block_size]

        mean_corrs = []

        for p in range(pitch_min, pitch_max + 1):

            # преобразуем блок сигнала в матрицу с числом столбцов p
            N = int(len(block) / p)
            b = block[:N * p].reshape((-1, p))

            # находим среднюю корреляцию строк
            correlations = [np.corrcoef(b[l], b[l + 1])[0, 1] for l in range(len(b) - 1)]
            if len(correlations) > 0:
                mean_corrs.append(np.mean(correlations))

        periodicity.append(np.max(mean_corrs))

    # участки из начала и конца сигнала не учитываем, т.к. там просто шум, вероятнее всего
    threshold = 0.6
    begin, end = 0, len(periodicity) - 1

    while periodicity[begin] < threshold:
        begin += 1
    while periodicity[end] < threshold:
        end -= 1

    # return np.mean(periodicity[begin:end+1]), periodicity
    return np.mean(periodicity[begin:end+1]) if begin < end else 0
