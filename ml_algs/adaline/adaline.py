import numpy as np

class Adaline:
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
    def losses(self):
        if not hasattr(self, 'losses_'):
            raise ValueError("Model is not fitted yet")
        
        return self.losses_

    def fit(self, X, y) -> 'Adaline':
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        np.random.seed(1)

        self.weights_ = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias_ = 0.0
        self.losses_ = []

        for _ in range(self.max_iter):
            pred = self.activation(self.net_input(X))
            errors = y - pred

            self.weights_ += self.learning_rate * 2.0 * (errors.dot(X) / X.shape[0])
            self.bias_ += self.learning_rate * 2.0 * errors.mean()

            loss_in_epoch = np.mean(errors ** 2)
            self.losses_.append(loss_in_epoch)

        return self
    
    def net_input(self, X) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
    
    def activation(self, X) -> np.ndarray:
        return X

    def predict(self, X) -> np.ndarray:
        if self.weights.size == 0:
            raise ValueError("Model is not fitted yet")
        
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    