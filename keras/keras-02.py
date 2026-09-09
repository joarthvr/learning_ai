import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5, 6])

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=7500)

# 4. 평가 예측
loss = result1 = model.evaluate(x, y)
print(loss)
result1 = model.predict(np.array([1, 2, 3, 4, 5, 6, 7]))
print('7', result1)
