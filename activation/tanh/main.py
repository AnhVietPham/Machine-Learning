import numpy as np
from math import exp

sigma = np.tanh


def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def activation_tanh(input, W, b):
    return sigma(W.dot(input) + b)


def activation_tanh_from_scratch(input, W, b):
    # 3x2 . 2x3
    input = np.expand_dims(input, axis=1)
    if W.shape[1] != input.shape[0]:
        print('No same size')
        return
    result = np.zeros((W.shape[0], input.shape[1]))
    for i in range(len(W)):  # Row of W
        for j in range(len(input[0])):  # Column of input
            for k in range(len(input)):  # Row of input
                result[i][j] += W[i][k] * input[k][j]
    result = result.T.squeeze(axis=0)
    for i in range(len(result)):
        result[i] = tanh(result[i] + b[i])
    return result


if __name__ == "__main__":
    # take a 3x3 matrix
    A = [[12, 7, 3],
         [4, 5, 6],
         [7, 8, 9]]

    # take a 3x4 matrix
    B = [[5, 8, 1, 2],
         [6, 7, 3, 0],
         [4, 5, 9, 1]]
    W = np.array([[-2, 4, -1], [6, 0, -3]])
    b = np.array([0.1, -2.5])

    # Input
    X = np.array([0.3, 0.4, 0.1])
    print(activation_tanh(X, W, b))
    print(activation_tanh_from_scratch(X, W, b))
