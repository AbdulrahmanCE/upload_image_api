import os

import skimage
import tensorflow as tf
import numpy as np
import pickle
import cv2
from os import listdir

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.core import Activation, Flatten, Dropout, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


# import matplotlib.pyplot as plt


def creat_model():
    EPOCHS = 1
    INIT_LR = 1e-3
    width = 256
    height = 256
    depth = 3

    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    ###################################################
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    ####################################################
    # convolution has size 4D
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    ####################################################
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    ####################################################
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    ####################################################
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    ####################################################
    # convert matrix 4D or 3D? to 2D
    model.add(Flatten())
    # build the lyers: 1024 nodes and has size 2D
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    model.summary()

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # distribution
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def convert_image_to_array(image_dir):
    default_image_size = tuple((256, 256))
    try:
        print('   Before the Resizing ')
        image = skimage.io.imread(image_dir)
        cv2.imwrite(image_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if image is not None:
            print('     After the Resizing ')
            image = cv2.resize(image, default_image_size)
            plt.imshow(image)
            plt.show()
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def predict(img_path):
    img_path = img_path[1:]
    # classifierLoad = tf.keras.models.load_model('cnn_model200.h5')
    # test_image = convert_image_to_array(img_path)
    #
    # test_image = np.expand_dims(test_image, axis=0)
    # result = classifierLoad.predict(test_image)
    #
    # if result[0][0] == 1:
    #     prediction = " 3 Healthy"
    # elif result[0][2] == 1:
    #     prediction = "5 iron"
    # else:
    #     prediction = "  4 other"
    #
    # print(result)
    # print(prediction)
    print(img_path)
    os.remove(img_path)
    return 'prediction'
