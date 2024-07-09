import torch

class SimpleNet:
    def __init__(self, input_len: int, output_len: int, regularization_rate: float, 
                 learning_rate: float, random_seed: int = 1):
        g = torch.Generator().manual_seed(random_seed)
        self.W = torch.randn(input_len, output_len, generator=g, requires_grad=True)
        self.W.grad = None
        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate
        self.losses = []

    def train(self, xs, ys, n_epochs: int):
        for _ in range(n_epochs):
            probs = self._forward(xs)

            loss = self.calculate_loss(probs, ys)
            loss.backward()
            
            print(loss.item())
            self.losses.append(loss.item())

            self.W.data += - self.learning_rate * self.W.grad

    def calculate_loss(self, probs, ys):
        return -probs[torch.arange(len(ys)), ys].log().mean() + self.regularization_rate * (self.W**2).mean() 
    
    def eval(self, xs):
        return self._forward(xs)

    def _forward(self, xs):
        logits = xs @ self.W
        counts = logits.exp()
        probs = counts / counts.sum(axis=1, keepdim=True)
        return probs