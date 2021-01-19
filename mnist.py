from sys import exit
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
import time
#
show_4 = not True
pathToData = 'C://Users//user//Desktop//STUDIYNG//КГ//mnistBinData//' # G:/AM/НС/mnist/
img_rows = img_cols = 28
num_classes = 10
epochs = 20
#
pathToHistory = '' # G:/AM/Лекции/
suff = '.txt'
# Имена файлов, в которые сохраняется история обучения
fn_loss = pathToHistory + 'loss_' + suff
fn_acc = pathToHistory + 'acc_' + suff
fn_val_loss = pathToHistory + 'val_loss_' + suff
fn_val_acc = pathToHistory + 'val_acc_' + suff
#
def show_x(x, img_rows, img_cols):
    print(x[0].shape)
    for k in range(1, 5):
        plt.subplot(2, 2, k)
        # Убираем 3-е измерение
        plt.imshow(x[k].reshape(img_rows, img_cols), cmap = 'gray')
        plt.axis('off')
    plt.show()
#
# Вывод графиков
def one_plot(n, y_lb, loss_acc, val_loss_acc):
    plt.subplot(1, 2, n)
    if n == 1:
        lb, lb2 = 'loss', 'val_loss'
        yMin = 0
        yMax = 1.05 * max(max(loss_acc), max(val_loss_acc))
    else:
        lb, lb2 = 'acc', 'val_acc'
        yMin = min(min(loss_acc), min(val_loss_acc))
        yMax = 1.0
    plt.plot(loss_acc, color = 'r', label = lb, linestyle = '--')
    plt.plot(val_loss_acc, color = 'g', label = lb2)
    plt.ylabel(y_lb)
    plt.xlabel('Эпоха')
    plt.ylim([0.95 * yMin, yMax])
    plt.legend()
#
def loadBinData(pathToData, img_rows, img_cols):
    print('Загрузка данных из двоичных файлов...')
    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_train = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
        y_train = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_test = np.fromfile(read_binary, dtype = np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_test = np.fromfile(read_binary, dtype = np.uint8)
    # Преобразование целочисленных данных в float32 и нормализация; данные лежат в диапазоне [0.0, 1.0]
    x_train = np.array(x_train, dtype = 'float32') / 255
    x_test = np.array(x_test, dtype = 'float32') / 255
    print(x_test.shape)
    x_train_shape_0 = int(x_train.shape[0] / (img_rows * img_cols))
    x_test_shape_0 = int(x_test.shape[0] / (img_rows * img_cols))
    x_train = x_train.reshape(x_train_shape_0, img_rows, img_cols, 1) # 1 - оттенок серого цвета
    x_test = x_test.reshape(x_test_shape_0, img_rows, img_cols, 1)
    # Преобразование в категориальное представление: метки - числа из диапазона [0, 9] в двоичный вектор размера num_classes
    # Так, в случае MNIST метка 5 (соответствует классу 6) будет преобразована в вектор [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    #print(y_train[0]) # (MNIST) Напечатает: 5
    print('Преобразуем массивы меток в категориальное представление')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    #print(y_train[0]) # (MNIST) Напечатает: [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    return x_train, y_train, x_test, y_test
#
# Определяем форму входных данных
input_shape = (img_rows, img_cols, 1)
# Создание модели нейронной сети
inp = Input(shape = input_shape) # Входной слой
x = inp
x = Reshape([-1], input_shape=input_shape)(x)
x = Dense(units = 32, activation = 'sigmoid')(x) # полносвзяный слой с 32 узлами и функцией активации relu
output = Dense(num_classes, activation = 'softmax')(x)
model = Model(inputs = inp, outputs = output)
model.summary()
loss = keras.losses.kullback_leibler_divergence # 'mse'
optimizer = keras.optimizers.Adam() # 'Adam'
#tf.keras.Model.compile принимает три важных аргумента:
##optimizer: Этот объект определяет процедуру обучения. Передайте в него экземпляры оптимизатора из модуля tf.keras.optimizers, 
#такие как tf.keras.optimizers.Adam или tf.keras.optimizers.SGD. Если вы просто хотите использовать параметры по умолчанию,
# вы также можете указать оптимизаторы ключевыми словами, такими как 'adam' или 'sgd'.
##loss: Это функция которая минимизируется в процессе обучения. Среди распространенных вариантов среднеквадратичная ошибка (mse), 
#categorical_crossentropy, binary_crossentropy. Функции потерь указываются по имени или передачей вызываемого объекта из модуля tf.keras.losses.
##metrics: Используются для мониторинга обучения. Это строковые имена или вызываемые объекты из модуля tf.keras.metrics.
#Кроме того, чтобы быть уверенным, что модель обучается и оценивается eagerly, проверьте что вы передали компилятору параметр run_eagerly=True
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
#
# Загрузка обучающего и проверочного множества из бинарных файлов
# Загружаются изображения и их метки
x_train, y_train, x_test, y_test = loadBinData(pathToData, img_rows, img_cols)
if show_4:
    show_x(x_test, img_rows, img_cols)
    exit()
#
start = time.time()
# Обучение нейронной сети
#tf.keras.Model.fit принимает три важных аргумента:
#epochs: Обучение разбито на *эпохи*. Эпоха это одна итерация по всем входным данным (это делается небольшими партиями).
#batch_size: При передаче данных NumPy, модель разбивает данные на меньшие блоки (batches) и итерирует по этим блокам во время обучения. Это число указывает размер каждого блока данных. Помните, что последний блок может быть меньшего размера если общее число записей не делится на размер партии.
#validation_data: При прототипировании модели вы хотите легко отслеживать её производительность на валидационных данных. Передача с этим аргументом кортежа входных данных и меток позволяет модели отображать значения функции потерь и метрики в режиме вывода для передаваемых данных в конце каждой эпохи.
history = model.fit(x_train, y_train, batch_size = 128, epochs = epochs,
                        verbose = 2, validation_data = (x_test, y_test))
time7 = time.time() - start
print('Время обучения: %f' % time7)
# Запись истории обучения в текстовые файлы
history = history.history
with open(fn_loss, 'w') as output:
    for val in history['loss']: output.write(str(val) + '\n')
with open(fn_acc, 'w') as output:
    for val in history['accuracy']: output.write(str(val) + '\n')
with open(fn_val_loss, 'w') as output:
    for val in history['val_loss']: output.write(str(val) + '\n')
with open(fn_val_acc, 'w') as output:
    for val in history['val_accuracy']: output.write(str(val) + '\n')
# Вывод графиков обучения
plt.figure(figsize = (9, 4))
plt.subplots_adjust(wspace = 0.5)
one_plot(1, 'Потери', history['loss'], history['val_loss'])
one_plot(2, 'Точность', history['accuracy'], history['val_accuracy'])
plt.suptitle('Потери и точность')
plt.show()


