from __future__ import print_function

# Computer Vision imports
import cv2
import numpy as np

# Machine Learning imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Training dataset
from keras.datasets import cifar10

def convert_img_array_gray(img_array):
    return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_array])

batch_size = 32
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
y_train, y_test = cifar10.load_data()
# Just get the images (not the names)
y_train, y_test = y_train[0], y_test[0]

# Build x data as grayscale versions
x_train, x_test = convert_img_array_gray(y_train), convert_img_array_gray(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert the target data to 1 dimensional arrays
y_train, y_test = np.reshape(y_train, (len(y_train), 32*32*3)), np.reshape(y_test, (len(y_test), 32*32*3))

model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten(input_shape=(32,32)))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(units=32*32*3, activation='sigmoid', input_shape=(32,32,1)))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('colorization_model.h5')
