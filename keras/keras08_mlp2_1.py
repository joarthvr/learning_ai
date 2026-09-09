import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1 데이터
x = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ]
)

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = x.transpose()
# print(x)

# 2. 모델구성
model = Sequential()
model.add(Dense(12, input_dim=3))
model.add(Dense(14))
model.add(Dense(16))
model.add(Dense(18))
model.add(Dense(16))
model.add(Dense(14))
model.add(Dense(12))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3)

# 4. 평가 예측
loss = model.evaluate(x, y)
results = model.predict(np.array([[10, 1.3, 0]]))  # (1,2)
print(loss)
print(results)

# 0.00011157737753819674
# [[10.006001]]
