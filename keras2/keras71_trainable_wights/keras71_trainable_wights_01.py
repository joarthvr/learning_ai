import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

print(tf.__version__)

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()

# ----------------------------------- model.weights ----------------------------------
print(model.weights)

# ---------------------------- model.trainable_weights -------------------------------
print(len(model.weights))  # ? 6
print(len(model.trainable_weights))  # ? 6

# ----------------------------------- 동결 ---------------------------------------------

model.trainable = False  #! 중요!  동결
print(len(model.weights))  # ? 6
print(len(model.trainable_weights))  # ? 0
# ----------------------------------- 동결 ---------------------------------------------

# ----------------------------------- model.weights ----------------------------------
print(model.weights)
