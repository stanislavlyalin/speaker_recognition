from scipy.io.wavfile import read
import numpy as np

def make_data_set(files, train_count, callback):
    
    # +1 - это один файл для тестирования
    records_count = train_count + 1

    train_set = np.array([])
    test_set = np.array([])

    # группировка по столбцу 1 - ID диктора
    for user, user_files_ids in files.groupby(1).groups.items():

        # пропускаем пользователей, у которых мало голосовых записей
        if len(user_files_ids) < records_count:
            continue
        
        for user_file_id in user_files_ids[:records_count]:

            file_path = files.iloc[user_file_id, 0]
            sample_rate, samples = read(file_path)

            features = callback(samples, sample_rate)
            sample = np.hstack((features, user_file_id))

            if user_file_id == user_files_ids[records_count-1]:
                test_set = np.hstack((test_set, sample))
            else:
                train_set = np.hstack((train_set, sample))

        print('\rprocessed user %d' % user, end='')

    return train_set.reshape((-1, records_count)), test_set.reshape((-1, records_count))
