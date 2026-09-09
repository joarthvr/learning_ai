import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])

x_test = np.array([8, 9, 10])
y_test = np.array([8, 9, 10])


# 2. 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(24))
model.add(Dense(36))
model.add(Dense(48))
model.add(Dense(36))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4)

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(np.array([11]))  # 추론 예측

print(loss)
print(result)

# 0.000001110626468915143
model.summary()
