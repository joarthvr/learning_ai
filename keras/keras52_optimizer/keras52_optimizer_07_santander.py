# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
import datetime
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

PATH = 'C:/study/_data/kaggle_santander/'
PATH = './_data/kaggle_santander/'

# 1. 데이터
train = pd.read_csv(PATH + 'train.csv', index_col=0)
test = pd.read_csv(PATH + 'test.csv', index_col=0)
submission = pd.read_csv(PATH + 'sample_submission.csv', index_col=0)

print(train.shape, test.shape, submission.shape)      # (200000, 201) (200000, 200) (200000, 1)
print(train.info(), test.info()) 

print(np.unique(train.target, return_counts=True))  # (array([0, 1]), array([179902,  20098]))

print(train.isna().sum())
print(test.isna().sum())

x = train.drop(columns='target')
y = train.target

print(x.shape, y.shape)  # (200000, 200) (200000,)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, stratify=y)

# scaler = MinMaxScaler().fit(x_train)
# scaler = StandardScaler().fit(x_train)
# scaler = MaxAbsScaler().fit(x_train)
scaler = RobustScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

PATH = './_save/keras33/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = f'{PATH}k31_{date}-{filename}'


# 2. 모델 구성
model = Sequential()
model.add(Dense(256, input_dim=200, activation='relu'))
model.add(Dropout(0.4))
# model.add(Dense(800, activation='relu'))
# model.add(Dense(800, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
learning_rate = 0.0005

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(patience=30, restore_best_weights=True)
mcp = ModelCheckpoint(filepath = filepath,
                      verbose=1,
                      save_best_only=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=2048, validation_split = 0.2, callbacks=[es])
end_time = time.time()
print(f'소요시간 : {end_time - start_time:.4f}')


# 4. 예측, 평가
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
print(f'learning_rate = {learning_rate}')
print(f'loss = {loss[0]:.4f}')
print(f'acc = {loss[1]:.4f}')
acc = accuracy_score(y_test, y_pred)
print(f'sklearn_acc = {acc:.4f}')


# CPU vs GPU
# GPU : 소요시간 : 63.5410
# CPU : 소요시간 : 39.0727


# epochs=50, batch_size=32, validation_split = 0.2
# EarlyStopping = X   => 과적합 추정
# loss = 0.2654
# acc = 0.9077

# batch_size늘리면 과적합이 금방 옴 (128, 512 시도하였음)

# epochs=50, batch_size=32, validation_split = 0.2
# patience=7, restore_best_weights=True
# loss = 0.2431
# acc = 0.9101

# epochs=50, batch_size=32, validation_split = 0.2
# patience=10, restore_best_weights=True
# loss = 0.2432
# acc = 0.9103

# loss = 0.2439
# acc = 0.9106

"""
MinMaxscaler 적용 후

loss = 0.2339
acc = 0.9130

StandardScaler 
loss = 0.2420
acc = 0.9108
sklearn_acc = 0.9108

MaxAbsScaler
loss = 0.2403
acc = 0.9113
sklearn_acc = 0.9113

RobustScaler
loss = 0.2413
acc = 0.9107
sklearn_acc = 0.9107

Save
loss = 0.2405
acc = 0.9108
sklearn_acc = 0.9108

load
loss = 0.2405
acc = 0.9108
sklearn_acc = 0.9108

dropout
loss = 0.2401
acc = 0.9105
sklearn_acc = 0.9105

learning_rate = 0.0005
loss = 0.2415
acc = 0.9105
sklearn_acc = 0.9105
"""

# # 5. Submit
# submit = model.predict(test)
# submission.target = np.round(submit)
# submission.to_csv(PATH + 'submit/submit_0908_1633.csv')

# result = pd.read_csv(PATH + 'submit/submit_0908_1633.csv', index_col=0)
# print(np.unique(result.target, return_counts=True))  # (array([0., 1.]), array([194064,   5936]))

# # 6. 시각화
# plt.plot(hist.history['loss'], label='loss')
# plt.plot(hist.history['val_loss'], label='val_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.title('Santander_EarlyStopping')
# plt.show()

