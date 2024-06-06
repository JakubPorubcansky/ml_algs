import pytest
import numpy as np
from sklearn.datasets import make_classification

from ml_algs.adaline import AdalineMiniBatchGradientDescent

def test_fit_predict_linearly_separable_data():
    p = AdalineMiniBatchGradientDescent(max_iter=20, learning_rate=0.05)

    train_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_y = np.array([0, 1, 1, 1])

    test_x = np.array([[-1, -1], [-1, 0], [0, -1], [2, 0], [0, 2], [2, 2]])
    test_y = np.array([0, 0, 0, 1, 1, 1])

    p.fit(train_X, train_y)

    train_prediction = p.predict(train_X)
    test_prediction = p.predict(test_x)

    assert np.all(train_prediction == train_y)
    assert np.all(test_prediction == test_y)

def test_fit_predict_not_linearly_separable_data():
    p = AdalineMiniBatchGradientDescent(max_iter=30, learning_rate=0.05)

    train_X, train_y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                            n_clusters_per_class=1, n_classes=2, random_state=10)

    min_expected_accuracy = 0.9

    p.fit(train_X, train_y)

    train_prediction = p.predict(train_X)

    assert np.sum(train_prediction == train_y) / len(train_y) >= min_expected_accuracy


def test_predict_if_not_fitted():
    p = AdalineMiniBatchGradientDescent()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    with pytest.raises(ValueError):
        p.predict(X)

