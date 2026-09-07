#11-3카피
from sklearn.datasets import fetch_california_housing, load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#2. 모델 설정
model = Sequential()
model.add(Dense(3,input_dim=13))
model.add(Dense(2))
# model.add(Dense(12))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4)

#3. 결과 예측
loss = model.evaluate(x_test, y_test)

print(loss)
# 31.57851219177246
# 43.3833770751953
# 51.372291564941406
# 52.74597930908203
# 69.20356750488281
# 87.4746322631836
# 202.54290771484375