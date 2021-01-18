import numpy as np

y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])


def mse(predictions, targets):
    differences = predictions - targets
    print("Differences: " + str(["%.8f" % elem for elem in differences]))
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    print("mean_of_differences_squared: " + str(mean_of_differences_squared))
    mse_val = np.sqrt(mean_of_differences_squared)
    return mse_val


print("d is: " + str(["%.8f" % elem for elem in y_hat]))
print("p is: " + str(["%.8f" % elem for elem in y_true]))
mse_val = mse(y_hat, y_true)
print("rms error is: " + str(mse_val))
