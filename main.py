import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Network:
    def __init__(self):
        self.training_data, self.test_data, self.training_labels, self.test_labels = self.get_moon_data()
        self.input_size = self.training_data.shape[0]
        self.input_dimensions = self.training_data.shape[1]
        self.hidden_node_count = 4
        self.output_node_count = 1
        self.weights, self.bias = \
            self.initialize_weights(self.input_dimensions, self.output_node_count, self.hidden_node_count)
        self.output1, self.activation1, self.output2,  self.activation2 = 0, 0, 0, 0

    def elu(self, input, a=2):
        # a is a hyperparameter
        # if input <= 0, return a * (np.exp(input) - 1), otherwise return input
        return np.where(input <= 0, a * (np.exp(input) - 1), input)

    def derivative_elu(self, input, a=2):
        return np.where(input <= 0, a * np.exp(input), 0)

    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))

    def get_moon_data(self):
        np.random.seed(0)
        X, Y = make_moons(500, noise=0.1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=73)
        return X_train, X_test, Y_train, Y_test

    def initialize_weights(self, input_dimensions, output_node_count, hidden_node_count):
        he = np.sqrt(3 / (0.5 * (input_dimensions + output_node_count)))  # Xavier Initialization
        weights, bias = [None] * 3, [None] * 3
        weights[1] = np.random.uniform(low=-he, high=he, size=(input_dimensions, hidden_node_count))
        bias[1] = np.zeros((1, hidden_node_count))
        weights[2] = np.random.uniform(low=-he, high=he, size=(hidden_node_count, output_node_count))
        bias[2] = np.zeros((1, output_node_count))
        return weights, bias

    def forward_propagation(self):
        self.output1 = np.dot(self.training_data, self.weights[1]) + self.bias[1]
        self.activation1 = self.elu(self.output1)
        self.output2 = np.dot(self.activation1, self.weights[2])
        self.activation2 = self.sigmoid(self.output2)  # network output
        return self.activation2

    def calculate_loss(self, output):
        labels = self.training_labels.reshape(-1, 1)
        log_probability = (np.multiply(np.log(output), labels) + np.multiply(np.log(1 - output), (1 - labels)))
        cross_entropy_loss = -1 / self.input_size * np.sum(log_probability)
        return cross_entropy_loss

    def back_propagation(self):
        m = 1/self.input_size   # regularizes the backprop step
        derivative_output2 = self.activation2 - self.training_labels.reshape(-1, 1)
        derivative_weights2 = m * np.dot(self.activation1.T, derivative_output2)
        derivative_bias2 = m * np.sum(derivative_output2, axis=0, keepdims=True)
        derivative_output1 = m * np.dot(derivative_output2, self.weights[2].T) * self.derivative_elu(self.output1)
        derivative_weights1 = m * np.dot(self.training_data.T, derivative_output1)
        derivative_bias1 = m * np.sum(derivative_output1, axis=0, keepdims=True)

        learning_rate = 0.01
        self.weights[2] -= learning_rate * derivative_weights2
        self.bias[2] -= learning_rate * derivative_bias2
        self.weights[1] -= learning_rate * derivative_weights1
        self.bias[1] -= learning_rate * derivative_bias1


network = Network()
network_output = network.forward_propagation()
loss = network.calculate_loss(network_output)

for i in range(1, 50000):
    network.back_propagation()
    network_output = network.forward_propagation()
    loss = network.calculate_loss(network_output)
    if i % 5000 == 0:
        print("%" + str(int(i/500)))
print("Final Loss: " + str(loss))





