import numpy as np

class Perceptron:
    def __init__(self, max_iter: int = 100, learning_rate: float = 0.001) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    @property
    def weights(self):
        if not hasattr(self, 'weights_'):
            raise ValueError("Model is not fitted yet")
        
        return self.weights_
    
    @property
    def bias(self):
        if not hasattr(self, 'bias_'):
            raise ValueError("Model is not fitted yet")
        
        return self.bias_
    
    @property
    def errors(self):
        if not hasattr(self, 'errors_'):
            raise ValueError("Model is not fitted yet")
        
        return self.errors_

    def fit(self, X, y) -> 'Perceptron':
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.weights_ = np.zeros(X.shape[1])
        self.bias_ = 0.0
        self.errors_ = []

        for _ in range(self.max_iter):
            n_errors_in_epoch = 0

            for x_i, y_i in zip(X, y):
                pred_i = self.predict(x_i)
                update = self.learning_rate * (y_i - pred_i)
                self.weights_ += update * x_i
                self.bias_ += update

                if update != 0:
                    n_errors_in_epoch += 1

            self.errors_.append(n_errors_in_epoch)

        return self
    
    def predict(self, X) -> np.ndarray:
        return np.where(np.dot(X, self.weights) + self.bias >= 0.0, 1, 0)
    