import numpy as np
import matplotlib.pyplot as plt

from Deep_Neural_Network.activation_function import Softmax_Activation
from Deep_Neural_Network.loss import Activation_Softmax_Loss_Categorical_Cross_Entropy
from Deep_Neural_Network.optimizer import Adam_Optimizer
import pickle


class Layer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.activation_function = activation_function

        self.inputs = None
        self.output = None

        self.d_weights = None
        self.d_biases = None
        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, d_values):
        # Gradients on parameters
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        # Gradient on values
        self.d_inputs = np.dot(d_values, self.weights.T)


class DNN:
    def __init__(self):
        self.layers = []
        self.acc_list = []
        self.loss_list = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, inputs):
        for idx, layer in enumerate(self.layers[:-1]):
            if idx == 0:
                layer.forward(inputs)
            else:
                layer.forward(self.layers[idx - 1].activation_function.output)

            layer.activation_function.forward(layer.output)

        self.layers[-1].forward(self.layers[-2].activation_function.output)

    def backward_propagation(self, loss_activation):
        self.layers[-1].backward(loss_activation.d_inputs)

        for idx, layer in reversed(list(enumerate(self.layers[: -1]))):
            layer.activation_function.backward(self.layers[idx + 1].d_inputs)
            layer.backward(layer.activation_function.d_inputs)

    def predict(self, x):
        self.forward_propagation(x)
        output = self.layers[-1].output

        softmax = Softmax_Activation()
        softmax.forward(output)

        return np.argmax(softmax.output)

    def train(self, X, y, batch_size=512, epochs=50):

        loss_activation = Activation_Softmax_Loss_Categorical_Cross_Entropy()
        optimizer = Adam_Optimizer()

        for epoch in range(epochs):

            current_idx = 0
            num_inputs = len(X)

            while current_idx + batch_size < num_inputs:

                X_batch = X[current_idx: current_idx + batch_size]
                y_batch = y[current_idx: current_idx + batch_size]

                self.forward_propagation(X_batch)
                loss = loss_activation.forward(self.layers[-1].output, y_batch)

                # Calculate accuracy from output of activation2 and targets
                # calculate values along first axis
                predictions = np.argmax(loss_activation.output, axis=1)
                if len(y_batch.shape) == 2:
                    y_batch = np.argmax(y_batch, axis=1)

                accuracy = np.mean(predictions == y_batch)
                if epoch % 5 == 0 and current_idx == 0:
                    print(f'epoch: {epoch}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f}, ' +
                          f'lr: {optimizer.current_learning_rate}')

                    self.acc_list.append(accuracy)
                    self.loss_list.append(loss)

                # Backward pass
                loss_activation.backward(loss_activation.output, y_batch)
                self.backward_propagation(loss_activation)

                # Update weights and biases
                optimizer.pre_update_params()
                for layer in self.layers:
                    optimizer.update_params(layer)

                optimizer.post_update_params()

                current_idx += batch_size

    def show_stats(self):
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        num_samples = len(self.acc_list)

        axs[0].set_title('accuracy')
        axs[0].plot(range(num_samples), self.acc_list, color='g')

        axs[1].set_title('loss')
        axs[1].plot(range(num_samples), self.loss_list, color='r')

        plt.show()

    def store_dnn(self, filename):
        pickle_file = open(filename, 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()

    def load_dnn(self, filename):
        pickle_file = open(filename, 'rb')
        loaded = pickle.load(pickle_file)
        pickle_file.close()

        return loaded
