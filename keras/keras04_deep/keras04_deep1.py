import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

# 2. 모델 생성
model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(24, input_dim=12))
model.add(Dense(12, input_dim=24))
model.add(Dense(1, input_dim=12))


# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000)

# 4. 평가, 예측.
loss = model.evaluate(x, y)
print('loss: ', loss)
result1 = model.predict(np.array([1, 2, 3, 4, 5, 6, 7]))
print(result1)
