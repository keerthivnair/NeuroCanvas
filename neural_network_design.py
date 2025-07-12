import numpy as np
import math
import json

class OCRNeuralNetwork:
    NN_FILE_PATH = 'ocr_network.json'  # Default file path

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, train_indices, verbose=False, use_file=False):
        self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
        self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
        self.hidden_layer_bias = self._rand_initialize_weights(1, 10)

        self.data = data_matrix
        self.labels = data_labels
        self.train_indices = train_indices
        self.verbose = verbose
        self.LEARNING_RATE = 0.1
        self._use_file = use_file

    def _rand_initialize_weights(self, size_in, size_out):
        return ((np.random.rand(size_out, size_in) * 0.12) - 0.06)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return np.multiply(sig, 1 - sig)

    def train_single_sample(self, x, label):
        y0 = np.asmatrix(x).T  # shape (400, 1)
        z1 = np.dot(np.asmatrix(self.theta1), y0) + np.asmatrix(self.input_layer_bias)
        y1 = self.sigmoid(z1)

        z2 = np.dot(np.array(self.theta2), y1) 
        z2 = np.add(z2,self.hidden_layer_bias)
        y2 = self.sigmoid(z2)

        actual_vals = [0] * 10 
        actual_vals[label] = 1
        output_errors = np.asmatrix(actual_vals).T - np.asmatrix(y2)
        hidden_errors = np.multiply(np.dot(np.asmatrix(self.theta2).T, output_errors), 
                                        self.sigmoid_prime(z1))

        # Gradient descent updates
        self.theta2 += self.LEARNING_RATE * np.dot(output_errors, y1.T)
        self.hidden_layer_bias += self.LEARNING_RATE * output_errors

        self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, y0.T)
        self.input_layer_bias += self.LEARNING_RATE * hidden_errors

        if self.verbose:
            prediction = np.argmax(y2)
            print(f"Predicted: {prediction}, Actual: {label}")

    def predict(self, test):
        
        print("👀 test type:", type(test))
        print("✅ test len:", len(test))
        print("🔢 first 10:", test[:10])
        
        y0 = np.asmatrix(test).T  # shape (400, 1)
        print("📐 y0 shape:", y0.shape)
        print("📐 theta1 shape:", self.theta1.shape)
        print("📐 input_layer_bias shape:", self.input_layer_bias.shape)
        
        z1 = np.dot(self.theta1, y0) + np.asmatrix(self.input_layer_bias)
        y1 = self.sigmoid(z1)

        z2 = np.dot(self.theta2, y1) + np.asmatrix(self.hidden_layer_bias)
        y2 = self.sigmoid(z2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }

        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        try:
            with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
                nn = json.load(nnFile)

            self.theta1 = np.array(nn['theta1'])
            self.theta2 = np.array(nn['theta2'])
            self.input_layer_bias = np.array(nn['b1'])
            self.hidden_layer_bias = np.array(nn['b2'])
        except:
            print("No saved model found. Starting fresh.")