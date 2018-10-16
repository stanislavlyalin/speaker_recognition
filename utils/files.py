import os
import glob
import re
import pandas as pd

# в указанной директории dir находит все wav-файлы и возвращает их в виде
# списка структур с полями {filepath, user_id, gender, age, record_id}
def files(dir):
    gender_dict = {'male': 0, 'female': 1}
    age_dict = {'youth': 0, 'adult': 1, 'senior': 2}

    wav_files = glob.glob(os.path.join(dir, '') + '*.wav')
    filelist = []

    for file in wav_files:
        try:
            res = re.findall('usr(\d{4})_(\w+)_(\w+)_(\d+)', file)[0]
            user_id = int(res[0])
            gender = gender_dict[res[1]]
            age = age_dict[res[2]]
            sample_id = int(res[3])
            filelist.append([file, user_id, gender, age, sample_id])
        except:
            pass

    return pd.DataFrame(filelist)
