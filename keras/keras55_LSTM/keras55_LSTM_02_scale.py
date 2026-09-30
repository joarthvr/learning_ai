import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array(
    [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [10, 11, 12],
        [20, 30, 40],
        [30, 40, 50],
        [40, 50, 60],
    ]
)
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x_pred = np.array([50, 60, 70])  # --> 80 맞춰보아요.79.5에 맞추기

x = x.reshape(-1, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)
print(x.shape, y.shape)

# 2. 모델 구성
model = Sequential()
model.add(
    SimpleRNN(10, input_shape=(3, 1), activation='relu')
)  # 중간 은닉층에 signmoid에 넣을 수 있음
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(patience=150, restore_best_weights=True)
model.fit(x, y, epochs=10000, callbacks=[es], validation_split=0.2)

# 4. 예측, 평가
y_pred = model.predict(x_pred)
print(y_pred)

# [[79.92799]]
