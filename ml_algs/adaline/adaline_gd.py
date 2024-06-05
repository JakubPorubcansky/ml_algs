import numpy as np

class AdalineGradientDescent:
    def __init__(self, max_iter: int = 100, learning_rate: float = 0.001) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self._weights = None
        self._bias = None

        self._losses = None

    @property
    def weights(self):
        if self._weights is None:
            raise ValueError("Model is not fitted yet")
        
        return self._weights
    
    @property
    def bias(self):
        if self._bias is None:
            raise ValueError("Model is not fitted yet")
        
        return self._bias
    
    @property
    def losses(self):
        if self._losses is None:
            raise ValueError("Model is not fitted yet")
        
        return self._losses

    def fit(self, X, y) -> 'AdalineGradientDescent':
        self._validate_before_fit(X, y)
        self._set_seed(seed=1)
        self._initialize_attributes(n_features=X.shape[1])

        for _ in range(self.max_iter):
            self._fit_one_epoch(X, y)

        return self
    
    def _validate_before_fit(self, X, y) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
    
    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)

    def _initialize_attributes(self, n_features: int) -> None:
        self._weights = np.random.normal(loc=0.0, scale=0.01, size=n_features)
        self._bias = 0.0
        self._losses = []

    def _fit_one_epoch(self, X, y) -> None:
        pred = self.activation(self.net_input(X))
        errors = y - pred

        self._weights += self.learning_rate * 2.0 * (errors.dot(X) / X.shape[0])
        self._bias += self.learning_rate * 2.0 * errors.mean()

        loss_in_epoch = np.mean(errors ** 2)
        self._losses.append(loss_in_epoch)

    def _update_parameters(self, x_i, y_i, pred_i) -> None:
        update = self.learning_rate * (y_i - pred_i)
        self._weights += update * x_i
        self._bias += update

    def net_input(self, X) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
    
    def activation(self, X) -> np.ndarray:
        return X

    def predict(self, X) -> np.ndarray:
        if self.weights.size == 0:
            raise ValueError("Model is not fitted yet")
        
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    