import numpy as np


class ReLU_Activation:
    def __init__(self):
        self.inputs = None
        self.d_inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inputs <= 0] = 0


class Softmax_Activation:
    def __init__(self):
        self.inputs = None
        self.d_inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.d_inputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.d_inputs[index] = np.dot(jacobian_matrix, single_dvalues)
