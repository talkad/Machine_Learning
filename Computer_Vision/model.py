import numpy as np
import cv2 as cv
import PIL
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
import tensorflow as tf


# the model implemented by Convolution Neural Networks and Image Recognition
# (MPLClassifier or SVM should be enough for this project)
class Model:

    def __init__(self):
        # building a linear stack of layers with the sequential model
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                                activation='relu',
                                input_shape=(150, 150, 1)))  # the first input is the number of images
        # each one of them is 150*150 pixels in Gray-Scale (one channel)

        # convolutional layers
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (4, 4), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (4, 4), activation='relu'))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='sigmoid'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(2, activation='softmax'))  # number of classes at the end layer

    def train_model(self, counters):
        X_train = np.array([])
        y_train = np.array([])

        for i in range(0, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg')[:, :, 0]
            img = img.reshape(16950)
            X_train = np.append(X_train, [img])
            y_train = np.append(y_train, 1)

        for i in range(0, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg')[:, :, 0]
            img = img.reshape(16950)
            X_train = np.append(X_train, [img])
            y_train = np.append(y_train, 2)

        X_train = X_train.reshape(counters[0] + counters[1], 16950)
        self.model.compile(loss=categorical_crossentropy,
                           optimizer=tf.optimizers.SGD(lr=0.01),
                           metrics=['accuracy'])

        self.model.fit(X_train, y_train,
                       batch_size=20,
                       epochs=10,
                       verbose=1)
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv.imread('frame.jpg')[:, :, 0]
        img = img.reshape(16950)
        prediction = self.model.predict([img])

        return prediction[0]
