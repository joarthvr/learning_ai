import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

a = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ]
).T  # 의도 = (10,2) / .T는 내가 삽입한 것
print(a)

size = 5

x_test = np.array([[7, 3], [8, 2], [9, 1], [10, 0]]).reshape(-1, size - 1, 2)
print(x_test.shape)

# 선생님 제작 함수 사용
"""
print(a.shape)  # (10,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size+1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)  # (6, 5, 2)

x = bbb[:, :-1, :]
# x = bbb[:, :-1]

y = bbb[:, :-1, -1]
# y = bbb[:, :-1, 1]

print(x.shape, y.shape)
# (6, 4, 2) (6, 4)
"""


# 자체 응용 함수 사용
def split_data(dataset, size):
    x = []
    y = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        x.append(subset[:-1])
        y.append(subset[-1][-1])
    return np.array(x), np.array(y)


x, y = split_data(a, size)
print(x.shape, y.shape)  # (6, 4, 2) (6, 2)
print(x)

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(4, 2)))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련
learning_rate = 0.001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(patience=200, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=150, verbose=1)
model.fit(
    x,
    y,
    epochs=3000,
    callbacks=[es, rlr],
    validation_split=0.1,
)

# 4. 예측
y_pred = model.predict(x_test)
print(y_pred)

# [[-0.9830116]]
