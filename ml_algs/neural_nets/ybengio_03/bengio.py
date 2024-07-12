import torch
import torch.nn.functional as F

class BengioNet:
    def __init__(self, n_all_letters: int, n_context_letters: int, n_features: int, n_hidden:int, 
                 regularization_rate: float, learning_rate: float, random_seed: int = 1):
        g = torch.Generator().manual_seed(random_seed)
        self.C = torch.randn(n_all_letters, n_features, generator=g)
        self.W = torch.randn(n_context_letters * n_features, n_all_letters, generator=g)
        self.b = torch.randn(n_all_letters, generator=g)
        self.H = torch.randn(n_context_letters * n_features, n_hidden, generator=g)
        self.d = torch.randn(n_hidden, generator=g)
        self.U = torch.randn(n_hidden, n_all_letters, generator=g)

        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate
        self.losses = []

        self.parameters = [self.C, self.W, self.b, self.H, self.d, self.U]
        for p in self.parameters:
            p.requires_grad = True
            p.grad = None

    def train(self, X, Y, batch_size: int, n_epochs: int):
        for _ in range(n_epochs):
            idcs = torch.randint(0, X.shape[0], (batch_size,))
            Xbatch = X[idcs]
            Ybatch = Y[idcs]

            probs = self._forward(Xbatch)
            loss = self.calculate_loss(probs, Ybatch)
            
            for p in self.parameters:
                p.grad = None
                
            loss.backward()
            
            for p in self.parameters:
                p.data += -self.learning_rate * p.grad

            self.losses.append(loss.item())

    def calculate_loss(self, probs, Y):
        return -probs[torch.arange(len(Y)), Y].log().mean() #+ self.regularization_rate * (self.W**2).mean() 
    
    def eval(self, xs):
        return self._forward(xs)

    def _forward(self, X):
        embd = self.C[X]
        embd_view = embd.view(-1, self.H.shape[0])

        logits = torch.tanh(self.d + embd_view @ self.H) @ self.U + embd_view @ self.W + self.b
        probs = F.softmax(logits, dim=1)
        return probs