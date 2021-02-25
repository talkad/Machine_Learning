import numpy as np
from Deep_Neural_Network.activation_function import Softmax_Activation


class Loss_Categorical_Cross_Entropy:
    def __init__(self):
        self.d_inputs = None

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = None
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.d_inputs = -y_true / dvalues
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples


class Activation_Softmax_Loss_Categorical_Cross_Entropy:
    def __init__(self):
        self.activation = Softmax_Activation()
        self.loss = Loss_Categorical_Cross_Entropy()

        self.output = None
        self.d_inputs = None

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.d_inputs = dvalues.copy()
        # Calculate gradient
        self.d_inputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples
