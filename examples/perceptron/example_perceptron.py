from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from ml_algs.perceptron.perceptron import Perceptron

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                        n_clusters_per_class=1, n_classes=2, random_state=1)

p = Perceptron(max_iter=100, learning_rate=0.001)

p.fit(X, y)

plt.figure(figsize=(10, 6))

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.gca().axline([0, - p.bias / p.weights[1]], [- p.bias / p.weights[0], 0], color='r')

plt.show()
