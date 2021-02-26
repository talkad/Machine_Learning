import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from Deep_Neural_Network.activation_function import ReLU_Activation
from Deep_Neural_Network.dnn import DNN, Layer, Dropout_Layer
from Deep_Neural_Network.loss import Activation_Softmax_Loss_Categorical_Cross_Entropy


def pre_process_data(trainx, trainy, testx, testy):
    # Normalize and flatten
    trainx = trainx / 255.
    testx = testx / 255.

    flat_trainx = [np.array([x for img_row in sub_list for x in img_row]) for sub_list in trainx]
    flat_testx = [np.array([x for img_row in sub_list for x in img_row]) for sub_list in testx]

    enc = OneHotEncoder(sparse=False, categories='auto')
    trainy = enc.fit_transform(trainy.reshape(len(trainy), -1))
    testy = enc.transform(testy.reshape(len(testy), -1))

    return np.array(flat_trainx), np.argmax(trainy, axis=1), np.array(flat_testx), np.argmax(testy, axis=1)


if __name__ == "__main__":
    # load MNIST digits dataset

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # print(f'Shape: X={train_X.shape}, y={train_y.shape}')
    #
    # for i in range(9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    #
    # plt.show()
    #
    # train_X, train_y, test_X, test_y = pre_process_data(train_X, train_y, test_X, test_y)
    #
    # # create the model
    # dnn = DNN()
    #
    # # hidden layers
    # dnn.add_layer(Layer(28*28, 32, ReLU_Activation(), Dropout_Layer(0.1)))
    # dnn.add_layer(Layer(32, 32, ReLU_Activation(), Dropout_Layer(0.1)))
    #
    # # output layer
    # dnn.add_layer(Layer(32, 10, Activation_Softmax_Loss_Categorical_Cross_Entropy(), None))
    #
    # dnn.train(train_X, train_y)
    #
    # dnn.show_stats()
    # dnn.store_dnn('DNN_32_32_dropout')

    dnn = DNN()
    loaded_dnn = dnn.load_dnn('DNN_32_32_dropout')
    loaded_dnn.show_stats()

    # while True:
    #     idx = int(input("Enter a sample index (0-3999): "))
    #     if idx < 0 or idx >= 4000:
    #         break
    #
    #     plt.title(f'sample {idx}')
    #     plt.imshow(test_X[idx], cmap=plt.get_cmap('gray'))
    #     plt.show()
    #
    #     flatten_x = [x for img_row in test_X[idx] for x in img_row]
    #     prediction = loaded_dnn.predict(flatten_x)
    #
    #     print(f'The prediction is {prediction} while the real answer is {test_y[idx]}')

    # check accuracy over the test samples
    num_accurate = 0
    idx_mistake = []
    pred_mistake = []
    real = []

    for i in range(len(test_y)):
        flatten_x = [x for img_row in test_X[i] for x in img_row]
        prediction = loaded_dnn.predict(flatten_x)

        if prediction == test_y[i]:
            num_accurate += 1
        else:
            idx_mistake.append(i)
            pred_mistake.append(prediction)
            real.append(test_y[i])

    print(f'accuracy percentage is {num_accurate / len(test_y) * 100}%')

    # print(f'the samples the model got wrong are: {idx_mistake}')
    #
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(test_X[idx_mistake[0]], cmap=plt.get_cmap('gray'))
    # axs[0, 0].set_title(f'pred. {pred_mistake[0]} vs. real {real[0]}')
    # axs[0, 1].imshow(test_X[idx_mistake[1]], cmap=plt.get_cmap('gray'))
    # axs[0, 1].set_title(f'pred. {pred_mistake[1]} vs. real {real[1]}')
    # axs[1, 0].imshow(test_X[idx_mistake[5]], cmap=plt.get_cmap('gray'))
    # axs[1, 0].set_title(f'pred. {pred_mistake[5]} vs. real {real[5]}')
    # axs[1, 1].imshow(test_X[idx_mistake[10]], cmap=plt.get_cmap('gray'))
    # axs[1, 1].set_title(f'pred. {pred_mistake[10]} vs. real {real[10]}')
    #
    # plt.show()
    #
