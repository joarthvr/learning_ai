import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, SimpleRNN
from tensorflow.keras.models import Sequential

# 1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = np.array(
    [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
    ]
)  # 시간 상 데이터 수기로 분리 [8,9,10] 은 학습시킬 데이터가 없음으로 생략

y = np.array([4, 5, 6, 7, 8, 9, 10])
print(x.shape, y.shape)  # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)  # 시계열 데이터는 3차원 필요 =
print(x.shape)  # (7, 3, 1)

# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1))) # 행 무시 열 우선
model.add(Bidirectional(SimpleRNN(10), input_shape=(3, 1)))
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

exit()
# 3. 컴파일, 훈련
es = EarlyStopping(monitor='loss', patience=150, restore_best_weights=True)
rlr = ReduceLROnPlateau(
    monitor='loss',
    verbose=1,
    patience=20,
)
model.compile(loss='mse', optimizer='adam')
model.fit(
    x,
    y,
    epochs=10000,
    #   callbacks=[es, rlr],
)

# 4. 평가, 예측
results = model.evaluate(x, y)
print(f'loss : {results:.4f}')

x_pred = np.array([8, 9, 10]).reshape(1, 3, 1)
y_pred = model.predict(x_pred)

print(f'[8,9,10]의 결과 : {y_pred}')
