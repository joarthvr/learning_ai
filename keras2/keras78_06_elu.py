import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1)


def elu(x, alpha):
    return (x > 0) * x + (x < 0) * (alpha * (np.exp(x) - 1))


y = elu(x, 2)
plt.plot(x, y)
plt.grid()
plt.show()
