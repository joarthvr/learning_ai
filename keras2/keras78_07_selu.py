import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1)

ALPHA = 1.6732632423543772
LAMBDA = 1.0507009873554805


def selu(x, ALPHA, LAMBDA):
    return LAMBDA * ((x > 0) * x + (x <= 0) * (ALPHA * (np.exp(x) - 1)))


y = selu(x, ALPHA, LAMBDA)
plt.plot(x, y)
plt.grid()
plt.show()
