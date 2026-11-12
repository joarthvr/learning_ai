import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1)


def silu(x):
    return x * (1 / 1 + np.exp(-x))


y = silu(x)
plt.plot(x, y)
plt.grid()
