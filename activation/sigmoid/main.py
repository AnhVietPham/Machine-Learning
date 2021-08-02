import numpy as np
import matplotlib.pyplot as plt


def sigmoid_activation(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_der(x):
    return x * (1 - x)


if __name__ == "__main__":
    z = np.linspace(-2.5, 2.5, 2000)
    plt.figure(figsize=(8, 4))
    plt.plot(z, sigmoid_activation(z), linewidth=1)
    plt.plot(z, sigmoid_der(sigmoid_activation(z)), linewidth=1)
    plt.grid(True)
    plt.legend(['Sigmoid', 'Sigmoid derivative'], loc='upper left')
    plt.title("Sigmoid Activation", fontsize=14)
    plt.show()
