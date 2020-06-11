import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# a naive logisic regression, really really naive
class MyLogisticRegression():

    def __init__(self, learning_rate, max_iter, batch_size, epsilon=0.0001, lambda_normalization=0.01):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.normalization = lambda_normalization

    def fit(self, X=None, y=None):
        '''
        1. Optimization:
            SGD
        2. Loss
            loss = log(y * sigmoid(k) + (1 - y) * (1 - sigmoid(k)))
                where k = WA+b
            normalization: L2
        3. Gradient:
            partial{loss}/partial{W} = k * (1 - k) / (y * sigmoid(k) + (1 - y) * (1 - sigmoid(k))) * (2 * y - 1) * transpose(A)
            partial{loss}/partial{b} = k * (1 - k) / (y * sigmoid(k) + (1 - y) * (1 - sigmoid(k))) * (2 * y - 1)
            + normalization
        4. Stop
            for simplicity, just iter for max_iter times.
        '''
        assert(X.shape[0] == y.shape[0])
        y = y.to_numpy().reshape(-1, 1)
        size, dim = X.shape

        self.weight = (np.random.rand(dim, 1) - 0.5) / math.sqrt(dim)
        self.bias = 0
        for i in range(self.max_iter):
            current_pose = 0
            while True:
                # get batch
                current_X = X[current_pose : current_pose + self.batch_size]
                current_y = y[current_pose : current_pose + self.batch_size]
                # get loss
                k = np.matmul(current_X, self.weight) + self.bias
                a = sigmoid(k)
                m = current_y * a + (1 - current_y) * (1 - a) + self.epsilon
                loss = (- current_y * np.log(a) - (1 - current_y) * np.log(1 - a)).mean()
                # calculate gradient
                common_part = (a - current_y)
                gradient_W = np.matmul(current_X.transpose(), common_part) / self.batch_size
                gradient_b = common_part.mean()
                # add normalization term
                gradient_W += self.weight * (2 * self.normalization)
                gradient_b += self.bias * (2 * self.normalization)
                # do weight clipping
                gradient_W = np.clip(gradient_W, -0.2, 0.2)
                gradient_b = np.clip(gradient_b, -0.2, 0.2)
                # bp
                self.weight = self.weight - gradient_W * self.learning_rate
                self.bias = self.bias - gradient_b * self.learning_rate
                # switch batch
                current_pose += self.batch_size
                if current_pose >= size:
                    break

    def predict(self, X=None):

        k = np.matmul(X, self.weight) + self.bias
        sigmoid_k = sigmoid(k)
        predicted = np.digitize(sigmoid_k, [0.2])
        return predicted