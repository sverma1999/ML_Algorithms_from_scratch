import numpy as np


class Logistic_regression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # Signmoid method
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # loss= -1/N SUMMATION(yi*log(yi_hat + epsilon) + (1-yi)*log(1-yi_hat + epsilon))
    def loss(self, y_true, y_pred):
        epsilon = 1e-9
        return -np.mean(
            y_true * np.log(y_pred + epsilon)
            + (1 - y_true) * np.log(1 - y_pred + epsilon)
        )

    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(z)
        return A

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias

        # gradient descent: looks like= w = w - lr * dw, b = b - lr * db
        # where dw = (1/n_samples) * X.T * (y_pred - y)
        # and db = (1/n_samples) * sum(y_pred - y)
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            y_pred = self.feed_forward(X)
            dz = y_pred - y

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # y_hat = np.dot(X, self.weights) + self.bias
        # y_pred = self._sigmoid(y_hat)
        y_pred = self.feed_forward(X)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]

        return np.array(y_pred_cls)

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
