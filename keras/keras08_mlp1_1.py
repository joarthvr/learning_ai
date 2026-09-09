import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
# x = np.array([[1,2,3,4,5],[6,7,8,10]])
x = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
y = np.array([1, 2, 3, 4, 5])

print(x.shape)  # (5,2)
print(y.shape)  # (5,)

# 2. 모델 구성
# 행 무시 열 우선
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000, batch_size=3)

# 4. 평가 예측
loss = model.evaluate(x, y)
results = model.predict(np.array([[6, 11]]))  # (1,2)
print(loss)
print(results)
