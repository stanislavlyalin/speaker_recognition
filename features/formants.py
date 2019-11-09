# coding: utf-8

import sys
import numpy as np
from .common import spectrogram, bin_to_hz
from statsmodels.stats.weightstats import DescrStatsW


# перевод частоты из герц в бины
def hz_to_bin(f, fft_size, sample_rate):
    return f * fft_size // sample_rate


# генератор кусочков спектрограммы
def sp_generator(sp, bandwidth, min_formants_duration):
    envelope = sp[:, :bandwidth].std(axis=1)
    total_energy = sum(envelope)
    epsilon = 1e-3
    env_sorted = sorted(envelope)

    # автоподбор порога
    for i in range(len(env_sorted)):
        energy = sum(env_sorted[:i]) + env_sorted[i] * (len(env_sorted) - i)

        # порог 0.54 получен по такой методике. Для группы фалов вручную были размечены
        # (с помощью графического планшета) интервалы, где присутствуют форманты.
        # Подбирался такой порог, чтобы автоматически вычисленные интервалы максимально
        # точно соответстовали ручной разметке. Более подробно описано в моей статье.
        if energy > 0.54 * total_energy:
            threshold = env_sorted[i] + epsilon
            break

    b, e = 0, 0
    for i in range(len(envelope) - 1):
        if envelope[i] < threshold and envelope[i+1] > threshold:
            b = i + 1
        elif envelope[i] > threshold and envelope[i+1] < threshold:
            e = i + 1

            if e - b + 1 >= min_formants_duration:
                yield sp[b:e, :], b, envelope[b:e+1].argmax(), envelope[b:e+1]


# функция поиска оптимальной дельты
def optimal_delta(sp_line, min_delta, max_delta):

    # строим график автокорреляционной функции
    c = np.correlate(sp_line, sp_line, mode='same')

    # через свёртку с окном в виде расчёски находим оптимальное значение delta
    best_conv = sys.float_info.min
    best_delta = min_delta

    for delta in range(min_delta, max_delta + 1):
        for i in range(min(delta * 5, len(c) - delta)):
            conv = np.sum(sp_line[i:i+delta * 5:delta])

            if conv > best_conv:
                best_conv = conv
                best_delta = delta

    return best_delta


# вычисление позиций для проверки с учётом текущих позиций и дельты
def make_positions(positions, step, min_delta, max_delta):
    diff = (np.arange(len(positions)) + 1) * step
    return positions + diff


# поиск наиболее вероятного положения формант относительно предыдущего
def best_position(sp_line, positions, min_delta, max_delta, formants_count):
    best_conv = []
    best_step = []

    for step in np.arange(-1, 1 + 1/formants_count, 1/formants_count):
        p = make_positions(positions, step, min_delta, max_delta)

        if min_delta <= p[0] <= max_delta:
            best_conv.append(np.sum(sp_line[p.round().astype(int)]))
            best_step.append(step)

    positions = make_positions(
        positions, best_step[np.argmax(best_conv)], min_delta, max_delta)

    return positions


# вычисление формантных признаков для файла
def formant_features(samples, sample_rate):

    # настройки
    min_df, max_df = 50, 250
    fft_size = 2048
    step_size = 256
    formants_count = 5
    bandwidth = formants_count * max_df
    bandwidth_in_bins = int(hz_to_bin(bandwidth, fft_size, sample_rate))
    min_formants_duration = 0.05  # в секундах, то есть 50 миллисекунд
    min_duration_in_lines = int(
        round(min_formants_duration / (step_size / sample_rate)))

    min_bin = hz_to_bin(min_df, fft_size, sample_rate)
    max_bin = hz_to_bin(max_df, fft_size, sample_rate)

    # вычисление спектрограммы
    sp = spectrogram(samples, fft_size, step_size)

    weights = []
    total_positions_hz = np.empty((0, formants_count))
    total_powers = np.empty((0, formants_count))
    variabilities = []

    for sp_piece, _, max_index, envelope_piece in sp_generator(sp, bandwidth_in_bins, min_duration_in_lines):

        weights += list(envelope_piece)

        best_delta = optimal_delta(
            sp_piece[max_index, :bandwidth_in_bins], min_bin, max_bin)
        positions = (np.arange(formants_count) + 1) * best_delta
        original_positions = positions.copy()
        piece_positions = np.empty((0, formants_count))

        # движемся вверх
        for i in range(max_index, -1, -1):
            positions = best_position(
                sp_piece[i], positions, min_bin, max_bin, formants_count)
            piece_positions = np.vstack((positions, piece_positions))
            total_powers = np.vstack(
                (total_powers, sp_piece[i, positions.astype(int)]))

        positions = original_positions.copy()

        # движемся вниз
        for i in range(max_index, len(sp_piece)):
            positions = best_position(
                sp_piece[i], positions, min_bin, max_bin, formants_count)
            piece_positions = np.vstack((piece_positions, positions))
            total_powers = np.vstack(
                (total_powers, sp_piece[i, positions.astype(int)]))

        total_positions_hz = np.vstack(
            (total_positions_hz, bin_to_hz(piece_positions, fft_size, sample_rate)))

        # изменения ЧОТ (pitch) в герцах в секунду для этого кусочка
        pitch_hz = bin_to_hz(piece_positions[:, 0], fft_size, sample_rate)
        pitch_variability = np.sum(np.abs(np.diff(pitch_hz)))
        piece_duration = len(sp_piece) * step_size / sample_rate
        variabilities.append(pitch_variability / piece_duration)

    # feature 1 - средняя частота первой форманты (ЧОТ)
    if len(weights) > 0:
        mean_f = np.average(total_positions_hz, axis=0, weights=weights)[0]
    else:
        mean_f = 0

    # feature 2 - средние мощности 5 формант относительно мощности первой форманты
    if len(weights) > 0:
        mean_A = np.average(total_powers, axis=0, weights=weights)
        mean_A /= mean_A[0]
    else:
        mean_A = np.ones(formants_count)

    # feature 3 - изменчивость голоса (на сколько герц меняется частота основного тона за секунду)
    variability = np.mean(variabilities)

    # feature 4 - дисперсия частоты первой форманты
    var_f = total_positions_hz.var(axis=0)[0]

    # feature 5 - дисперсии мощностей формант, нормированные относительно первой форманты
    if len(weights) > 0:
        var_power = [DescrStatsW(
            total_powers[:, i], weights=weights).var for i in range(formants_count)]
        var_power /= var_power[0]
    else:
        var_power = np.zeros(formants_count)

    return np.hstack((mean_f, mean_A, variability, var_f, var_power))
