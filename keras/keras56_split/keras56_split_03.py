import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))  # 101~106까지 찾자.

size = 6

# loss는 0.1 이하.


def split_data(dataset, size, predict):
    x = []
    y = []
    x_pred = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        x.append(subset[:-1])
        y.append(subset[-1])
    for i in range(len(predict) - size + 2):
        subset = predict[i : (i + size - 1)]
        x_pred.append(subset)
    return (
        np.array(x).reshape(-1, size - 1, 1),
        np.array(y),
        np.array(x_pred).reshape(-1, size - 1, 1),
    )


x, y, x_pred = split_data(a, size, x_predict)
print(x.shape, y.shape, x_pred.shape)  # (95, 5, 1) (95,) (5, 5, 1)

# exit()

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(5, 1), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
learning_rate = 0.0001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(patience=300, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=30, verbose=1)
model.fit(
    x,
    y,
    epochs=3000,
    batch_size=32,
    callbacks=[es, rlr],
    validation_split=0.2,
)

# 4. 예측
y_pred = model.predict(x_pred)
print(y_pred)

# [[100.9886  ]
#  [102.01178 ]
#  [103.0087  ]
#  [104.01044 ]
#  [105.008675]
#  [105.9961  ]]
