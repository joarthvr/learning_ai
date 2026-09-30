import datetime
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

# acc = 0.93

# 1. 데이터
dataset = fetch_covtype(as_frame=True)
x = dataset.data
y = dataset.target

# print(x.shape, y.shape)  # (581012, 54) (581012,)
# print(x.head())
# print(y.head())
# print(np.unique(y, return_counts=True))   # (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301, 35754, 2747, 9493, 17367, 20510]))
# print(dataset.feature_names)
# print(dataset.DESCR)    # 54 features, 7 classes

# print(x.info(), y.info())  # no null

y = pd.get_dummies(y, dtype='int')
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3, stratify=y)

# scaler = MinMaxScaler().fit(x_train)
# scaler = StandardScaler().fit(x_train)
# scaler = MaxAbsScaler().fit(x_train)
scaler = RobustScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

PATH = "./_save/keras31/"
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = f'{PATH}k31_{date}-{filename}'


# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=54, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
learning_rate = 0.0001
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics = ['acc'])
es = EarlyStopping(patience=30, restore_best_weights=True)
mcp = ModelCheckpoint(filepath=filepath,
                      verbose=1,
                      save_best_only=True)
start = time.time()
hist = model.fit(x_train, y_train,
          epochs = 300,
          batch_size=256,
          callbacks = [es],
          validation_split = 0.2)
end = time.time()


# model = load_model(f'{PATH}k31_0914_1443-0109-0.1323.keras')
# 4. 예측, 평가
result = model.evaluate(x_test, y_test)
print(f'소요시간 = {end - start:.4f}')
print(f'learning_rate = {learning_rate}')
print(f'loss = {result[0]:.4f}')
print(f'acc = {result[1]:.4f}')
# print(f"val_loss = {hist.history['val_loss']}")
# print(f"val_acc = {hist.history['val_acc']:.4f}")
# print(f"loss = {hist.history['loss']:.4f}")
# print(f"acc = {hist.history['acc']:.4f}")

# CPU vs GPU
# GPU : 소요시간 = 80.6674
# CPU : 소요시간 = 76.6155


"""
epochs = 1000,
batch_size=256,
callbacks = patience 20,
validation_split = 0.2

acc: 0.9026
loss: 0.2490
"""

"""
epochs = 1000,
batch_size=128,
callbacks = patience 30,
validation_split = 0.2

소요시간 = 408.1546
loss = 0.1734
acc = 0.9336
"""

"""
MinMaxscaler 적용 후

소요시간 = 378.9729
loss = 0.1380
acc = 0.9504

StandardScaler
소요시간 = 804.7961
loss = 0.1296
acc = 0.9567

MaxAbsScaler
소요시간 = 828.1796
loss = 0.1367
acc = 0.9526

RobustScaler
소요시간 = 566.1155
loss = 0.1371
acc = 0.9553

Save
소요시간 = 407.9604
loss = 0.1333
acc = 0.9541

load
loss = 0.1333
acc = 0.9541

소요시간 = 175.8907
learning_rate = 0.01
loss = 0.2719
acc = 0.8946

learning_rate = 0.0001
loss = 0.1726
acc = 0.9362
"""
