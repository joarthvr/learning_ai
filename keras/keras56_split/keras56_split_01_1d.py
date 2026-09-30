import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

a = np.array(range(1, 11))
size = 5  # timestep 사이즈

print(a.shape)  # (10,)


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x(a, size)
print(bbb)
print(bbb.shape)  # (6, 5)

datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([7, 8, 9, 10]).reshape(-1, size - 1, 1)
print(x_test.shape)


def split_data(dataset, size):
    x = []
    y = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        x.append(subset[:-1])
        y.append(subset[-1])
    return np.array(x).reshape(-1, size - 1, 1), np.array(y)


x, y = split_data(a, size)
print(x, y)  # (6, 4, 1) (6,)

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(4, 1), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
learning_rate = 0.0001
rlr = ReduceLROnPlateau(patience=30, verbose=1)
es = EarlyStopping(patience=20, restore_best_weights=True)
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
model.fit(
    x,
    y,
    epochs=10000,
    callbacks=[es, rlr],
    validation_split=0.3,
)

# 4. 예측, 평가
y_pred = model.predict(x_test)
print(y_pred)

# [[11.018449]]
