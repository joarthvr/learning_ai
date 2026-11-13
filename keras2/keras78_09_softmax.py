import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 5)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


y = softmax(x)

ratio = y
labels = y

plt.pie(ratio, labels, shadow=True)
plt.grid()
plt.show()
