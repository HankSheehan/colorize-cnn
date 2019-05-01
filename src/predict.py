import cv2
import numpy as np


from keras.models import load_model

from keras.datasets import cifar10


model = load_model('hsv.h5')

test_set = cifar10.load_data()[0][0]

for test in test_set:

    cv2.imshow('original', test)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale', test)
    test = np.reshape(test, (1, 32, 32)).astype('float32')
    test = np.expand_dims(test, axis=-1)

    # TODO: THESE NEED TO BE CHANGED FOR THE RANGES OF HSV OR WHATEVER COLORSPACE
    test /= 255

    prediction = model.predict([test])[0]

    prediction = np.reshape(prediction, (32, 32, 3))

    prediction = cv2.cvtColor(prediction, cv2.COLOR_HSV2BGR)

    cv2.imshow('prediction', prediction)

    cv2.waitKey()
