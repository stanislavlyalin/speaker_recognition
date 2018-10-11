import os
import re
import glob
import tarfile


# удаляет все файлы в директории
def clear_dir(dir):
    for f in glob.glob(dir + '/*'):
        os.remove(f)


# создает директорию, если не существует, иначе очищает ее
def create_empty_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    clear_dir(dir)


# извлекает wav-файлы и описание из архива в указанный каталог
def extract_wav_and_description(tar_file, dest_dir):
    infos = []
    for item in tar_file.getmembers():
        if re.search('\.wav$', item.name) or re.search('README$', item.name):
            infos.append(item)
    for info in infos:
        info.name = os.path.basename(info.name)
        tar_file.extract(info, dest_dir)


# извлечение пола и возраста из текстового файла описания
def gender_and_age(filename):
    with open(filename) as f:
        content = f.readlines()
    gender, age = '', ''
    for line in content:
        g_res = re.findall('Gender: \[?(\w+)\]?', line)
        a_res = re.findall('Age [Rr]ange: \[?(\w+)\]?', line)
        if len(g_res) > 0:
            gender = g_res[0].lower()
        if len(a_res) > 0:
            age = a_res[0].lower()
    return gender, age


os.chdir('voices')
files = os.listdir()
print('directory has %d archives' % len(files))

# группировка дикторов и их голосовых записей
users = {}
for file in files:
    result = re.findall('(.*)-\d+.*', file)
    if len(result) > 0:
        user = result[0]
        if user != 'anonymous':
            if user in users.keys():
                users[user].append(file)
            else:
                users[user] = [file]

# создание каталогов для временных файлов и результатов
temp_dir = 'temp'
dest_dir = '../samples'
readme = temp_dir + '/README'
create_empty_dir(temp_dir)
create_empty_dir(dest_dir)

# цикл извлечения файлов из архивов
user_counter = 0
total_users = len(users)

for user in users.keys():
    user_counter += 1

    file_counter = 0
    for file in users[user]:
        tgz = tarfile.open(file)
        extract_wav_and_description(tgz, temp_dir)

        if not os.path.isfile(readme):
            continue

        gender, age = gender_and_age(readme)

        for tmp_wav in glob.glob(temp_dir + '/*.wav'):
            file_counter += 1
            dest_filename = 'usr%04d_%s_%s_%03d.wav' % (user_counter, gender, age, file_counter)
            os.rename(tmp_wav, dest_dir + '/' + dest_filename)

        os.remove(readme)

    print('\rcomplete %.2f%%' % (user_counter * 100.0 / total_users), end='')

clear_dir(temp_dir)
os.removedirs(temp_dir)