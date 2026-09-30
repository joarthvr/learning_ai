import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))  # 101~106까지 찾자.

size = 6

# 데이터를 reshape한 후, split_x 함수로 시계열 데이터로 변환
# (N, 10, 1) -> (N, 5, 2)
# 결과를 뽑는다

print(a.shape)

a = a.reshape(-1, 2)
print(a)

x_pred = np.array(x_predict).reshape(-1, 5, 2)
print(x_pred)


def split_data(dataset, size):
    x = []
    y = []
    # x_pred = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        x.append(subset[:-1])
        y.append(subset[-1][-1])
    # for i in range(len(predict) - size+2):
    #     subset = predict[i:(i+size-1)]
    #     x_pred.append(subset[:,:,-1])
    return np.array(x), np.array(y)


x, y = split_data(a, size)

print(x.shape, y.shape)

exit()

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(5, 2), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
learning_rate = 0.0001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(patience=200, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=70, verbose=1)
model.fit(
    x,
    y,
    epochs=30000,
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

# [[107.58163]]
