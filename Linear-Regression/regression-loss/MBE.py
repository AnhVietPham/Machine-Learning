import numpy as np

y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])

print("y_hat is: " + str(["%.8f" % elem for elem in y_hat]))
print("y_true is: " + str(["%.8f" % elem for elem in y_true]))


def mbe(predictions, targets):
    differences = predictions - targets
    mean_of_difference = differences.mean()
    return mean_of_difference


mbe_val = mbe(y_hat, y_true)
print("Mean error is: " + str(mbe_val))
