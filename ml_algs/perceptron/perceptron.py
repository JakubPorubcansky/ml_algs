import numpy as np

class Perceptron:
    def __init__(self, max_iter: int = 100, learning_rate: float = 0.001) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self._weights = np.array([])
        self._bias = 0.0

    @property
    def weights(self):
        return self._weights
    
    @property
    def bias(self):
        return self._bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self._weights = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            prediction = self.predict(X)

            for x_i, y_i, pred_i in zip(X, y, prediction):
                update = self.learning_rate * (y_i - pred_i)
                self._weights += update * x_i
                self._bias += update

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._weights.size == 0:
            raise ValueError("Model is not fitted yet")
        
        return np.where(np.dot(X, self._weights) + self._bias >= 0.0, 1, 0)
    