import numpy as np
from sklearn.metrics import accuracy_score

# from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

#! 1. Data ---------------------------------------
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

#! 2. Model -------------------------------------
# model = Perceptron()
model = LinearSVC()

#! 3. 훈련 --------------------------------------
model.fit(x_data, y_data)

#! 4. 평가, 예측 -------------------------------
y_predict = model.predict(x_data)

result = model.score(x_data, y_data)

print('model.score:', result)

acc = accuracy_score(y_data, y_predict)

print('acc_score: ', acc)

"""
model.score: 0.5
acc_score:  0.5
"""
