# Speaker Recognition
Реализация методов идентификации диктора по голосу на Python.

## Структура проекта
### preprocess
Функции предобработки сигнала.

**normalize** - нормализация сигнала. Из каждого элемента вычитается среднее, затем разности делятся на среднеквадратическое отклонение.

**remove_silence** - удаление пауз. Модуль сигнала сглаживается окном заданного размера. Затем сохраняются только те отсчёты, для которых сглаженный массив оказался выше заданного порога.

### features
Функции извлечения признаков из речевых сигналов.

**lpc** - вычисление коэффициентов линейного предсказания. Будущий отсчёт можно выразить как сумму **_p_** предыдущих отсчётов, умноженных на коэффициенты **_a_**. Эти коэффициенты можно вычислить, решив задачу регрессии. Функция **lpc** принимает на вход отсчёты сигнала, частоту дискретизации, размер блока сигнала в семплах, на которые будет разбиваться исходный сигнал, количество рассчитываемых коэффициентов. Коэффициенты рассчитываются отдельно для каждого блока сигнала. Таким образом, получается таблица коэффициентов, число строк в которой соответствует числу блоков в сигнале, число столбцов соответствует числу признаков. Затем из таблицы формируется вектор средних значений коэффициентов по столбцам и вектор дисперсий. Эти два вектора и являются итоговым вектором признаков для данного файла.

**mfcc** - вычисление мел-частотных кепстральных коэффициентов. Функция принимает на вход отсчёты сигнала, частоту дискретизации и количество рассчитываемых коэффициентов. Как и в случае с **lpc**, сигнал разбивается на блоки, и коэффициенты MFCC считаются для каждого блока. В результате получается таблица с числом строк, соответствующим числу блоков сигнала, и числом столбцов, соответствующим количеству коэффициентов. Затем из таблицы формируется вектор средних значений коэффициентов по столбцам и вектор дисперсий. Эти два вектора и являются итоговым вектором признаков для данного файла.

### utils
Различные вспомогательные функции.

**files** - формирует список структур с информацией о голосовых записях в заданной директории. На вход принимает путь к директории. Возвращает список структур с полями ```{путь_к_файлу, id_диктора, пол_диктора, возраст_диктора, id_записи_этого_диктора}```.

**make_data_set** - формирует обучающую и тестовую выборку. Принимает на вход список файлов, полученный с помощью функции **files**, количество пользователей (на случай, если нужно обработать голосовые записи не всех пользователей из директории с файлами), количество записей каждого пользователя, которые идут на обучение, callback-функция, в которой будет выполняться расчёт признаков. **callback-функция** принимает отсчёты сигнала и частоту дискретизации и возвращает вектор признаков для данного файла.

**samples_to_data_sets** - разбивает исходный сигнал samples на блоки по data_set_len отсчётов, а затем в каждом блоке упорядочивает данные столбиком с n входными значениями и одним выходным. Функция полезна при формировании выборок для, например, коэффициентов линейного предсказания (**lpc**) или коэффициентов, полученных в результате обучения нейронной сети.

**mistakes_to_folders** - раскладывает по каталогам файлы, на которых классификатор ошибся.


# Сценарии использования

## Формирование обучающей и тестовой выборок
Для этого необходимо запустить скрипт `all_features.py`. Скрипт вычислит признаки и сохранит их в файлах `train_set.npy` и `test_set.npy`. Описание признаков приведено в файле `features_description.txt`.

## Раскладывание по каталогам файлов для ошибочных примеров
Если на признаках, которые были почерпнуты из публикаций и интернета, качество классификации недостаточно, нужно вручную проанализировать файлы, на которых классификатор ошибся с целью найти различия и представить их в виде признаков. Для этого удобно разложить файлы с правильным классом и файлы с неправильным классом, но предсказанным как правильный, по каталогам для последующего ручного анализа. Для этого используется функция `mistakes_to_folders` из `каталога `utils`. В качестве параметров принимает:
* model - модель (классификатор).
* X_test, y_test - тестовая выборка. Число признаков (столбцов) должно соответствовать обучающей выборке, на которой училась модель.
* wav_files - файлы, по которым формировалась обучающая и тестовая выборка.
* mistakes_dir - каталог, в который будет помещён результат - подкаталоги с файлами

Пример использования:

```python
import os
import pickle
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.mistakes_to_folders import mistakes_to_folders

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

# признаки, которые оказались значимы при тренировке модели
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

wav_files = glob.glob('i:/aspirantura/samples_16kHz_train_test_100/*.wav')
mistakes_dir = './mistakes_dir/'

mistakes_to_folders(model, X_test[:, indexes], y_test, wav_files, mistakes_dir)
```
