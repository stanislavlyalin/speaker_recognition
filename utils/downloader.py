# скрипт загружает данные с сайта (http://www.voxforge.org/home)
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import glob
import os

path = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'
web_page = urllib.request.urlopen(path).read()
doc = BeautifulSoup(web_page, 'html.parser')

# получение ссылок на файлы
links = np.array([])
for link in doc.select('pre a'):
    links = np.append(links, link.get('href'))
links = links[4:]

# получение списка уже загруженных файлов
downloaded_files = glob.glob('C:/work/projects/speaker_recognition/voices/*.tgz')
downloaded_files = [os.path.basename(file) for file in downloaded_files]
print('downloaded files: %d' % len(downloaded_files))

# получение имён файлов, которые ещё не загружены
links = np.setdiff1d(links, downloaded_files)
print('files to download: %d' % len(links))

# скачивание файлов с записями
counter = 0
size = len(links)
for link in links:
    urllib.request.urlretrieve(path + link, 'C:/work/projects/speaker_recognition/voices/' + link)
    counter += 1
    print('\r%.2f%%, remaining %d files' % (counter * 100 / size, size - counter), end='')
