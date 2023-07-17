from tensorflow.keras.datasets import mnist #Библиотека с базой Mnist
from tensorflow.keras.models import Sequential #Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense #Подключаем класс Dense - полносвязный слой
from tensorflow.keras.optimizers import Adam #Подключаем оптимизатор Adam
from tensorflow.keras import utils #Утилиты для to_categorical
from tensorflow.keras.preprocessing import image #Для отрисовки изображения
import numpy as np #работа с многомерными массивами
import pandas as pd #обработка и анализ данных/работа с таблицами
import pylab #Модуль для построения графиков
from mpl_toolkits.mplot3d import Axes3D #Модуль для трехмерной графики
import matplotlib.pyplot as plt #Отрисовка изображений
from PIL import Image #Отрисовка изображений


(x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data() #Загрузка данных Mnist
# x_train_org и y_train_org - данные обучающей выборки
# x_test_org и y_train_org - данные тестовой выборки

'''
Обучающую выборку мы используем для того, чтобы обучить сеть, 
в то время как тестовая используется для того, чтобы проверить, 
насколько качественно произошло это обучение.
Смысл тестовой выборки в том, чтобы проверить, насколько точно 
отработает наша сеть с данными, с которыми она не сталкивалась ранее. 
Именно это является самой важной метрикой оценки сети.
'''
#Меняем формат входных картинок с 28х28 на 784х1
#x_train_org - объект numpy, reshape - метод класса numpy
# 60000 - это количество картинок, т.е. 60000 двумерных массивов numpy, 784 - это 28х28=784, 784 элемента одномерного массива numpy
# применяя метод reshape , получаем 60000 одномерных массивов размером 784 элемента
x_train = x_train_org.reshape(60000, 784)
x_test = x_test_org.reshape(10000, 784)

# Преобразуем ответы в формат one_hot_encoding
y_train = utils.to_categorical(y_train_org, 10)
y_test = utils.to_categorical(y_test_org, 10)

#метод to_categorical - возвращает двоичное представление входных данных массива numpy
#первый аргумент - это сам элемент, в нашем случае 5, второй аргумент количество классов
#в нашем случае числа от 0 до 9. Значит классов 10.
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
utils.to_categorical(y_train_org[0], 10)

# Модель
model_sample = Sequential() # Создаём сеть прямого распространения
model_sample.add(Dense(800, input_dim=784, activation='relu')) # Добавляем полносвязный слой на 800 нейронов с relu-активацией
model_sample.add(Dense(10, activation='softmax')) # Добавляем полносвязный слой на 10 нейронов с softmax-активацией

model_sample.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Компилируем модель

# Обучение сети
# fit - функция обучения нейронки
# x_train, y_train - обучающая выборка, входные и выходные данные
# batch_size - размер батча, количество примеров, которое обрабатывает нейронка перед одним изменением весов
# epochs - количество эпох, когда нейронка обучается на всех примерах выборки
# verbose - 0 - не визуализировать ход обучения, 1 - визуализировать
model_sample.fit(x_train, y_train, batch_size=8, epochs=300, verbose=1, validation_split=0.2)

model_sample.evaluate(x_test, y_test)[1]

data = [[800, 'relu', 128, round(model_sample.evaluate(x_test, y_test, verbose = 0)[1], 3)]]

example = image.load_img('./numbers/num.png', target_size=(28, 28), color_mode = 'grayscale')

# Нарисуем картинку
# example.convert('RGBA') - библиотека PIL, класс Image, метод convert
# конвиртируем изображение в RGBA цветах
# https://pillow.readthedocs.io/en/stable/reference/Image.html
# matplotlib.pyplot.imshow библиотека matplotlib, пакет pyplot, метод imshow
# imshow - возвращает данные, как изображение
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
plt.imshow(example.convert('RGBA'))

# Нормализуем данные
example = image.img_to_array(example) # преобразуем изображение в numpy-массив
example = example.reshape(1,784)
example = example.astype('float32')
example = 1-example/255
example = np.where(example > 100, 255, example) # По условию меняем значения в масиве
#Распознаём наш пример
pred_example = model_sample.predict(example)
print(pred_example)

# Получаем индекс самого большого элемента (это итоговая цифра, которую распознала сеть)
pred = np.argmax(pred_example)
print(pred)
