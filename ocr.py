import numpy as np
from neural_network_design import OCRNeuralNetwork
from sklearn.datasets import fetch_openml
from skimage.transform import resize

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().reshape(-1,28,28)
y = mnist.target.astype(int)

X_resized = np.array([resize(img, (20,20),anti_aliasing=True) for img in X])
X_flat = X_resized.reshape(-1, 400)
X_flat = X_flat / X_flat.max()

X_small = X_flat[:400]
y_small = y[:400]

np.save("data_matrix.npy", X_small)
np.save("data_labels.npy", y_small)



data_matrix = np.load('data_matrix.npy')
data_labels = np.load("data_labels.npy")
train_indices = list(range(400))  

nn = OCRNeuralNetwork(15, data_matrix, data_labels, train_indices, verbose=True)

for idx in train_indices:
    x = data_matrix[idx]
    label = data_labels[idx]
    nn.train_single_sample(x, label)


nn.save()