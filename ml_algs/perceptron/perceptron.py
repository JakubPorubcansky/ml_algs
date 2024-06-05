import numpy as np

class Perceptron:
    def __init__(self, max_iter: int = 100, learning_rate: float = 0.001) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        self._weights = None
        self._bias = None

        self.n_errors_in_epoch = 0
        self._errors = None

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
    def errors(self):
        if self._errors is None:
            raise ValueError("Model is not fitted yet")
        
        return self._errors

    def fit(self, X, y) -> 'Perceptron':
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
        self._errors = []

    def _fit_one_epoch(self, X, y) -> None:
        self.n_errors_in_epoch = 0

        for x_i, y_i in zip(X, y):
            pred_i = self.predict(x_i)
            self._update_parameters(x_i, y_i, pred_i)

        self._errors.append(self.n_errors_in_epoch)

    def _update_parameters(self, x_i, y_i, pred_i) -> None:
        update = self.learning_rate * (y_i - pred_i)
        self._weights += update * x_i
        self._bias += update

        if update != 0:
            self.n_errors_in_epoch += 1

    def predict(self, X) -> np.ndarray:
        return np.where(np.dot(X, self.weights) + self.bias >= 0.0, 1, 0)
    