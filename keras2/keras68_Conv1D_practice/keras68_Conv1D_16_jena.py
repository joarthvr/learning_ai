# [학습 정리] keras68 Conv1D jena
# keras59(Bidirectional GRU) 의 jena wd 예측을 Conv1D 로 바꿔서 RNN 성능을 넘기는 실험.

import datetime
import time

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

PATH = './_data/kaggle_jena/'

# 2016.12.31의 wd값을 맞추기
# wd 열 드랍
# 2016.12.31 데이터 (144개) 제외하고 학습

path = './_save/keras68/'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = f'{path}k68_{date}-{filename}'

# 1. 데이터

df = pd.read_csv(PATH + 'jena_climate_2009_2016.csv', index_col=0)
print(df.shape)  # (420551, 14)

x = df.drop(columns=['wd (deg)'])
y = df['wd (deg)']

print(x.shape, y.shape)  # (420551, 13) (420551,)

x_data = x[:-288]
x_pred = x[-288:-144]
y_data = y[144:-144]
y_cor = y[-144:]

print(x_data.shape, y_data.shape)  # (420407, 13) (420407,)

size = 144


def split_data(datasets, target, size):
    x = []
    y = []
    for i in range(len(datasets) - size + 1):
        subset = datasets[i : (i + size)]
        x.append(subset)
        subset = target[i : (i + size)]
        y.append(subset)
    return np.array(x), np.array(y)


start_time = time.time()
x_data, y_data = split_data(x_data, y_data, size)
x_pred, y_cor = split_data(x_pred, y_cor, size)

print(x_data.shape, y_data.shape)  # (420120, 144, 13) (420120, 144)
print(x_pred.shape, y_cor.shape)  # (1, 144, 13) (1, 144)
end_time = time.time()

print(f'자르는 시간 = {end_time - start_time:.4f}')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42, test_size=0.3)

print(x_train.shape, x_test.shape)

x_train = x_train.reshape(-1, 13)
x_test = x_test.reshape(-1, 13)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred.reshape(-1, 13))

x_train = x_train.reshape(-1, 144, 13)
x_test = x_test.reshape(-1, 144, 13)
x_pred = x_pred.reshape(-1, 144, 13)

print(x_train.shape, x_test.shape)

# 2. 모델 구성
model = Sequential()
# padding='same' 으로 길이 144 를 유지하고, MaxPooling 으로 144 → 72 → 36 으로 줄인다.
# 커널 5 + dilation(1, 2, 4) 로 한 층이 보는 시간 범위를 넓힌다 (dilation 은 커널 사이를 건너뛰며 훑는 간격).
model.add(Conv1D(64, 5, padding='same', activation='relu', input_shape=(144, 13)))
model.add(Conv1D(64, 5, padding='same', dilation_rate=2, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 5, padding='same', dilation_rate=4, activation='relu'))
model.add(Conv1D(128, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, padding='same', activation='relu'))
# 시점의 위치도 정보라서 GlobalPooling 대신 Flatten 으로 (36, 128) 을 전부 넘긴다.
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(144))

print(f'파라미터 수 = {model.count_params():,}')

# 3. 컴파일, 훈련
learning_rate = 0.001
model.compile(
    loss='mse',
    optimizer=Adam(
        learning_rate=learning_rate,
    ),
)
es = EarlyStopping(
    patience=10,
    restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    patience=5,
    verbose=1,
)
mcp = ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)
start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=512,
    callbacks=[es, rlr, mcp],
    validation_split=0.25,
)
end_time = time.time()

# 4. 예측, 평가
y_pred = model.predict(x_pred)

print(y_cor.shape, y_pred.shape)
loss = model.evaluate(x_pred, y_cor)
print(f'소요시간 = {end_time - start_time:.4f}')
print(f'loss = {loss:.4f}')
print(f'rmse : {root_mean_squared_error(y_cor, y_pred):.4f}')

"""
[기준] keras59 Bidirectional GRU
소요시간 = 547.3360
loss = 3479.5752
rmse : 44.8387

소요시간 = 398.3425
loss = 3793.9871
rmse : 48.1634

[갱신]
소요시간 = 641.1806
loss = 3010.3804
rmse : 36.6358
"""
