from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

from ml_algs.perceptron import Perceptron
from ml_algs.adaline import Adaline

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                        n_clusters_per_class=1, n_classes=2, random_state=10)
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

a = Adaline(max_iter=20, learning_rate=0.05)
p = Perceptron(max_iter=20, learning_rate=0.001)

p.fit(X, y)
a.fit(X, y)

plt.figure(figsize=(15, 20))

plt.subplot(3, 1, 1)

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.gca().axline([0.0, - p.bias / p.weights[1]], [1.0, (- p.bias - p.weights[0]) / p.weights[1]], color='red', label='Perceptron')
plt.gca().axline([0.0, (0.5 - a.bias) / a.weights[1]], [1.0, (0.5 - a.bias - a.weights[0]) / a.weights[1]], color='blue', label='Adaline')

plt.xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
plt.ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)

plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(1, len(p.errors) + 1), p.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.title('Perceptron: Number of Misclassifications vs. Epochs')

plt.subplot(3, 1, 3)
plt.plot(range(1, len(a.losses) + 1), a.losses, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Adaline: Loss vs. Epochs')

plt.subplots_adjust(hspace=0.5)

plt.show()
