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
model.fit(x_train, y_train, epochs=100, batch_size=4,
        verbose=1,
        validation_split=0.30)

#3. 결과 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("R^2: ", r2_score(y_test, model.predict(x_test)))
print("RMSE: ", root_mean_squared_error(y_test, model.predict(x_test)))