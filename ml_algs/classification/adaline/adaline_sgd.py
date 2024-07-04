from typing import Generator
import numpy as np

class AdalineMiniBatchGradientDescent:
    def __init__(self, max_iter: int = 100, learning_rate: float = 0.001, random_seed: int = 1) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self._batch_size = None
        self._random_seed = random_seed

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

    def fit(self, X, y, batch_size: float | int = 0.2) -> 'AdalineMiniBatchGradientDescent':
        self._store_batch_size(batch_size, n_samples_total=X.shape[0])
        self._validate_before_fit(X, y)
        self._set_seed()
        self._initialize_attributes(n_features=X.shape[1])
 
        for _ in range(self.max_iter):
            self._fit_one_epoch(X, y)

        return self
    
    def _store_batch_size(self, batch_size: float | int, n_samples_total: int) -> None:
        if isinstance(batch_size, int):
            if batch_size <= 0 or batch_size > n_samples_total:
                raise ValueError("batch_size, if specified as integer, must be greater than 0 and less than or equal to the number of samples in X") 
            
            self._batch_size = batch_size

        elif isinstance(batch_size, float):
            if batch_size <= 0.0 or batch_size > 1.0:
                raise ValueError("batch_size, if specified as float, must be greater than 0.0 and less than or equal to 1.0")
            self._batch_size = np.ceil(batch_size * n_samples_total).astype(int)
    
    def _validate_before_fit(self, X, y) -> None:
        if X.shape[0] == 0:
            raise ValueError("X must have at least one sample")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
    
    def _set_seed(self) -> None:
        np.random.seed(self._random_seed)

    def _initialize_attributes(self, n_features: int) -> None:
        self._weights = np.random.normal(loc=0.0, scale=0.01, size=n_features)
        self._bias = 0.0
        self._losses = []

    def _fit_one_epoch(self, X, y) -> None:
        for X_batch, y_batch in self._generate_batches(X, y):
            self.fit_on_batch(X_batch, y_batch)
        
        loss_in_epoch = self._calculate_loss(X, y)
        self._losses.append(loss_in_epoch)


    def _generate_batches(self, X, y) -> Generator:
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for i in range(0, n_samples, self._batch_size):
            idx_until = min(i + self._batch_size, n_samples)
            batch_indices = indices[i:idx_until]
            yield X[batch_indices], y[batch_indices]
            
    def fit_on_batch(self, X, y):
        pred = self.activation(self.net_input(X))
        errors = y - pred

        self._weights += self.learning_rate * 2.0 * (errors.dot(X) / X.shape[0])
        self._bias += self.learning_rate * 2.0 * errors.mean()

    def _calculate_loss(self, X, y) -> float:
        pred = self.activation(self.net_input(X))
        errors = y - pred
        return np.mean(errors ** 2)

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
    