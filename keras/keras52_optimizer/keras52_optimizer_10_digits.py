import datetime
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# acc : 1.0

# 1. 데이터
dataset = load_digits(as_frame=True)

x = dataset.data
y = dataset.target

# print(x.shape, y.shape)     # (1797, 64) (1797,)
# print(dataset.feature_names)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))
# print(dataset.DESCR)    # 64 columns, 10 classes

y = pd.get_dummies(y)
print(y.shape)  # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=y)

# scaler = MinMaxScaler().fit(x_train)
# scaler = StandardScaler().fit(x_train)
# scaler = MaxAbsScaler().fit(x_train)
scaler = RobustScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


PATH = './_save/keras31/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = f'{PATH}k31_{date}-{filename}'


# 2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
learning_rate = 0.005
model.compile(
    loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc']
)
es = EarlyStopping(patience=250, restore_best_weights=True)
mcp = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
start = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=32, callbacks=[es], validation_split=0.2)
end = time.time()


# model = load_model(f'{PATH}k31_0914_1451-0017-0.2049.keras')

# CPU vs GPU
# GPU : 소요시간 = 8.9609
# CPU : 소요시간 = 9.4439

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print(f'loss = {result[0]:.4f}')
print(f'acc = {result[1]:.4f}')
print(f'소요시간 = {end - start:.4f}')
print(f'learning_rate = {learning_rate}')
y_pred = tf.one_hot(np.argmax(y_pred, axis=1), y_pred.shape[1])
# print(y_test.shape, y_pred.shape)
acc = accuracy_score(y_test, y_pred)
print(f'acc = {acc:.4f}')


# loss = 0.2376
# acc = 0.9356

"""
MinMaxscaler 적용 후

loss = 0.1918
acc = 0.9511

StandardScaler
loss = 0.2546
acc = 0.9378

MaxAbsScaler
loss = 0.1986
acc = 0.9578

RobustScaler
loss = 0.2219
acc = 0.9378

Save
loss = 0.2139
acc = 0.9444

load
loss = 0.2139
acc = 0.9444
''
learning_rate = 0.01
loss = 0.1866
acc = 0.9600

learning_rate = 1e-05
loss = 0.3417
acc = 0.9111

learning_rate = 0.005
loss = 0.1877
acc = 0.9511
"""
