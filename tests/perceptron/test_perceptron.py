import pytest
import numpy as np
from ml_algs.perceptron.perceptron import Perceptron

def test_fit_predict_linearly_separable_data():
    p = Perceptron(max_iter=10, learning_rate=0.001)

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
    p = Perceptron(max_iter=40, learning_rate=0.1)

    train_X = np.array([[0, 4], [1, 4], [2, 4], [3, 4], [4, 1], [0, 0], [1, 0], [2, 0], [3, 0], [4, 3]])
    train_y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    expected_y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])

    p.fit(train_X, train_y)

    train_prediction = p.predict(train_X)

    assert np.all(train_prediction == expected_y)


def test_predict_if_not_fitted():
    p = Perceptron()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    with pytest.raises(ValueError):
        p.predict(X)

