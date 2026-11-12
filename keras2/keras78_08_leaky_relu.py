import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1)

ALPHA = 0.01


def leaky_relu(x, ALPHA):
    # return np.maximum(ALPHA * x, x)
    return np.where(x > 0, x, ALPHA * 2)


y = leaky_relu(x, ALPHA)
plt.plot(x, y)
plt.grid()
plt.show()
