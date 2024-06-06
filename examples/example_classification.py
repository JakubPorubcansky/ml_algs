from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

from ml_algs.perceptron import Perceptron
from ml_algs.adaline import AdalineGradientDescent, AdalineMiniBatchGradientDescent
from ml_algs.logistic_regression import LogisticRegressionMiniBatchGradientDescent

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                        n_clusters_per_class=1, n_classes=2, random_state=10)

agd = AdalineGradientDescent(max_iter=30, learning_rate=0.05)
amgd50 = AdalineMiniBatchGradientDescent(max_iter=30, learning_rate=0.01)
amgd20 = AdalineMiniBatchGradientDescent(max_iter=30, learning_rate=0.01)
amgd1 = AdalineMiniBatchGradientDescent(max_iter=30, learning_rate=0.01)
le = LogisticRegressionMiniBatchGradientDescent(max_iter=30, learning_rate=0.5)
pcn = Perceptron(max_iter=30, learning_rate=0.001)

pcn.fit(X, y)
agd.fit(X, y)
amgd50.fit(X, y, batch_size=50)
amgd20.fit(X, y, batch_size=20)
amgd1.fit(X, y, batch_size=1)
le.fit(X, y, batch_size=0.2)

plt.figure(figsize=(15, 20))

plt.subplot(3, 1, 1)

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.gca().axline([0.0, -pcn.bias / pcn.weights[1]], [1.0, (-pcn.bias - pcn.weights[0]) / pcn.weights[1]], color='red', label='Perceptron')
plt.gca().axline([0.0, (0.5 - agd.bias) / agd.weights[1]], [1.0, (0.5 - agd.bias - agd.weights[0]) / agd.weights[1]], color='grey', label='AdalineGradientDescent')
plt.gca().axline([0.0, (0.5 - amgd50.bias) / amgd50.weights[1]], [1.0, (0.5 - amgd50.bias - amgd50.weights[0]) / amgd50.weights[1]], color='blue', label='AdalineMiniBatchGradientDescent(batch_size=50)')
plt.gca().axline([0.0, (0.5 - amgd20.bias) / amgd20.weights[1]], [1.0, (0.5 - amgd20.bias - amgd20.weights[0]) / amgd20.weights[1]], color='green', label='AdalineMiniBatchGradientDescent(batch_size=20)')
plt.gca().axline([0.0, (0.5 - amgd1.bias) / amgd1.weights[1]], [1.0, (0.5 - amgd1.bias - amgd1.weights[0]) / amgd1.weights[1]], color='lightgreen', label='AdalineMiniBatchGradientDescent(batch_size=1)')
plt.gca().axline([0.0, -le.bias / le.weights[1]], [1.0, (-le.bias - le.weights[0]) / le.weights[1]], color='orange', label='LogisticRegressionMiniBatchGradientDescent(batch_size=20)')

plt.xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
plt.ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)

plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(1, len(pcn.errors) + 1), pcn.errors, marker='o', color='red')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.title('Perceptron: Number of Misclassifications vs. Epochs')

plt.subplot(3, 1, 3)
plt.plot(range(1, len(agd.losses) + 1), agd.losses, marker='o', color='grey')
plt.plot(range(1, len(amgd50.losses) + 1), amgd50.losses, marker='o', color='blue')
plt.plot(range(1, len(amgd20.losses) + 1), amgd20.losses, marker='o', color='green')
plt.plot(range(1, len(amgd1.losses) + 1), amgd1.losses, marker='o', color='lightgreen')
plt.plot(range(1, len(le.losses) + 1), le.losses, marker='o', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')

plt.subplots_adjust(hspace=0.5)

plt.show()
