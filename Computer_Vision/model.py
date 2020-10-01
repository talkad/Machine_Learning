import cv2 as cv
import PIL
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

# the model implemented by Convolution Neural Networks and Image Recognition
# (MPLClassifier or SVM should be enough for this project)
from setuptools import glob


class Model:

    def __init__(self):
        # building a linear stack of layers with the sequential model
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=(150, 113, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        # create neural network of three layers: 2 of them with 16 nodes and the output layer with one node (0/1)
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

        self.model.summary()

    def train_model(self):
        # first try
        # X_train = np.array([])
        # y_train = np.array([])
        #
        # filenames = [img for img in glob.glob("img/*.jpg")]
        # shuffle(filenames)
        #
        # for filename in filenames:
        #     match = re.match(r'frame(\d+)class(\d+)', filename[4: -4])
        #     frame = int(match.group(1))
        #     class_num = int(match.group(2))
        #
        #     img = cv.imread(f'img/frame{frame}class{class_num}.jpg')
        #     img = img.reshape(150, 113, 3)
        #     X_train = np.append(X_train, [img])
        #     y_train = np.append(y_train, [class_num])
        #
        # X_train = X_train.reshape((counters[0] + counters[1], 150, 113, 3))
        # X_train = X_train / 255.0
        #
        # self.model.fit(X_train, y_train,
        #                batch_size=64,
        #                epochs=50,
        #                verbose=0)

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            'img',
            target_size=(150, 113),
            batch_size=16,
            class_mode='binary')

        self.model.fit(
            train_generator,
            # steps_per_epoch=100,
            epochs=50)

        self.model.save_weights('CNN.h5')
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        # cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        cv.imwrite("frame.jpg", frame)
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv.imread('frame.jpg')
        img = img.reshape(1, 150, 113, 3)

        img = img / 255.0

        prediction = self.model.predict([img])

        return prediction[0][0]
