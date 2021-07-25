import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1 - x)


if __name__ == "__main__":
    (X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
                        cluster_std=1.05, random_state=20)

    print("[INFO] starting training...")
    W = np.random.uniform(size=(X.shape[1],))

    lossHistory = []

    for epoch in np.arange(0, args["epochs"]):
        preds = sigmoid_activation(X.dot(W))

        error = preds - y

        loss = np.sum(error ** 2)
        lossHistory.append(loss)
        print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))

        gradient = X.T.dot(error) / X.shape[0]

        W += -args["alpha"] * gradient

    # Predict
    for i in np.random.choice(250, 10):
        activation = sigmoid_activation(X[i].dot(W))
        label = 0 if activation < 0.5 else 1

        # show our output classification
        print("activation={:.4f}; predicted_label={}, true_label={}".format(
            activation, label, y[i]))

    # compute the line of best fit by setting the sigmoid function
    # to 0 and solving for X2 in terms of X1
    Y = (-(W[0] * X)) / W[1]

    # plot the original data along with our line of best fit
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
    plt.plot(X, Y, "r-")

    # construct a figure that plots the loss over time
    fig = plt.figure()
    plt.plot(np.arange(0, args["epochs"]), lossHistory)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()
