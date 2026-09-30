import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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
model.add(LSTM(units=8, input_shape=(3, 1), return_sequences=True, activation='relu'))
model.add(LSTM(16, return_sequences=True, activation='relu'))
model.add(LSTM(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()


# [실습] 나머지 진행 (55-2 와 성능비교)


# 3. 컴파일, 훈련

learning_rate = 0.001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(patience=300, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=200, verbose=1)
model.fit(x, y, epochs=10000, callbacks=[es, rlr], validation_split=0.2)

# 4. 예측, 평가
y_pred = model.predict(x_pred)
print(y_pred)
