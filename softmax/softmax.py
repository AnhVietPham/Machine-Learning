import numpy as np
import torch


def softmax(Z):
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis=1, keepdims=True)
    return A


def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = e_Z / e_Z.sum(axis=1, keepdims=True)
    return A


if __name__ == "__main__":
    Z = np.array([[2, 2, 2], [2, 1.8, 0]])
    print(Z)
    score = softmax_stable(Z)
    print(score)
