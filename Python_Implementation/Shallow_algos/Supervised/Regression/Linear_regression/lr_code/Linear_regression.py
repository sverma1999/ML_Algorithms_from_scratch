import numpy as np


class Linear_regression:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        # Feature scaling
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # This will give the number of samples and the number of features e.g. (100, 2)
        num_samples, num_features = X_normalized.shape  # X shape [N, f]

        # Initialize the weights for the number of features
        self.weights = np.random.rand(num_features)  # X shape [f, 1]

        # Initialize the bias
        self.bias = 0

        # Gradient Descent
        prev_loss = float("inf")
        for i in range(self.iterations):
            # Hypothesis
            y_predicted = (
                np.dot(X_normalized, self.weights) + self.bias
            )  # X shape [N, f] * [f, 1] = [N, 1]

            # Compute derivative of the loss function with respect to the weights
            # X -> [N, f]
            # y_pred -> [N, 1]
            # dw -> [f, 1]
            # Note, we are doing X.T here becuase this will transpose the matrix X from [N, f] to [f, N] and allow us to multiply it by
            # the error vector (y_pred - y) which is of shape [N, 1]
            # [f, N] * [N, 1] = [f, 1]
            df_df = (1 / num_samples) * np.dot(X_normalized.T, (y_predicted - y))

            # Compute derivative of the loss function with respect to the bias
            df_db = (1 / num_samples) * np.sum(y_predicted - y)

            # w = w - learningRate * df/dw
            # b = b - learningRate * df/db

            self.weights -= self.learning_rate * df_df
            self.bias -= self.learning_rate * df_db

            # Compute loss
            loss = np.mean((y_predicted - y) ** 2)

            # Check stopping criterion
            if abs(loss - prev_loss) < 1e-6:
                break

            prev_loss = loss

        return self

    def predict(self, X):
        # Feature scaling
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return np.dot(X_normalized, self.weights) + self.bias
