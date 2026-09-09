import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# 8개, 4개, 4개로 자른다

x_train = x[:8]
y_train = y[:8]

x_val = x[8:12]
y_val = y[8:12]

x_test = x[12:]
y_test = y[12:]


# 2. 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(24))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_data=(x_val, y_val))

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
r2 = r2_score(y_test, model.predict(x_test))
rmse = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))


##################################################

print('loss : ', loss)
print('r2 : ', r2)
print('rmse : ', rmse)
