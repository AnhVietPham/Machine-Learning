import numpy as np
import matplotlib.pyplot as plt

# Gradient descent is an optimization algorithm used to minimize some function by iteratively
# moving in the direction of steepest descent as defined by the negative of the gradient.

x = 10
y = []
for i in range(10):
    x = x - 0.1 * 2 * x
    y.append(x ** 2)

plt.plot(y)
plt.xlabel('Số lần')
plt.ylabel('f(x)')
plt.title('Giá trị f(x) sau số lần thực hiện bước 2')
plt.show()
