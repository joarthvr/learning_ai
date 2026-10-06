# 55-2카피
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, GlobalAveragePooling1D
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
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3, 1)))
model.add(Conv1D(10, 2))
model.add(GlobalAveragePooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='loss', patience=150, restore_best_weights=True)
rlr = ReduceLROnPlateau(
    monitor='loss',
    verbose=1,
    patience=20,
)
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(patience=200, restore_best_weights=True)
model.fit(x, y, epochs=10000, callbacks=[es], validation_split=0.25)

# 4. 예측, 평가
y_pred = model.predict(x_pred)
print(y_pred)

# [[79.92799]]
