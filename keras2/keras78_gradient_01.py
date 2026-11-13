import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 6, 100)

print(x, len(x))


def f(x):
    return x**2 - 4 * x + 6


y = f(x)


plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk', color='red')
plt.grid()
plt.xlabel('x')
plt.xlabel('y')
plt.show()
