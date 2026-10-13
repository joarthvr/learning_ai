import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

print(tf.__version__)
#! 1. data ---------------------------------------------
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

#! 2. model ---------------------------------------------
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()

# ----------------------------------- 동결 ---------------------------------------------

model.trainable = False  #! 중요!  동결
# model.trainable = True  #! 중요!  동결

#! 3. compile & fit ---------------------------------------------
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=100, verbose=1)
