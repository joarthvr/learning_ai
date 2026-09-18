import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 4, 6])

# 2. 모델 생성
model = Sequential()
# 딥러닝 레이어
model.add(Dense(4, input_dim=1))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(1))


# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=4000)

# 4. 평가, 예측.
loss = model.evaluate(x, y)
print('loss: ', loss)
# result1 = model.predict(np.array([1,2,3,4,5,6,7]))
# print(result1)
