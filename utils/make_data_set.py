def make_data_set(files, train_count, callback):
    for user_files in files.groupby(1).groups:
        # вычитываем все семплы
        # вызываем callback, в который передаём samples и sample_rate
