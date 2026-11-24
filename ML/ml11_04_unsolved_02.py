import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# from sklearn.linear_model import Perceptron
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

tf.random.set_seed(333)
np.random.seed(333)

#! 1. Data ---------------------------------------
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

#! 2. Model -------------------------------------
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1, activation='sigmoid'))

#! 3. 컴파일, 훈련 --------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=1000)

#! 4. 평가, 예측 ---------------------------------------
y_predict = model.predict(x_data)

result = model.evaluate(x_data, y_data)

print('loss, acc', result)

acc = accuracy_score(y_data, np.round(y_predict))

print('acc_score: ', acc)

"""
loss, acc [0.6931471824645996, 0.5]
acc_score:  0.5
"""
