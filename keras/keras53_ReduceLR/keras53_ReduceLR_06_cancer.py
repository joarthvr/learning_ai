import datetime
import time

import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
# x = datasets['data']
y = datasets.target

print(x.shape, y.shape)  # (569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>

# 0과 1의 개수가 몇개인지 찾아보기. - numpy
print(np.unique(y))  # [0 1]
print(np.unique(y, return_counts=True))  # [0 1] (array([0, 1]), array([212, 357]))

# 0과 1의 개수가 몇개인지 찾아보기 -  pandas
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212

print(pd.Series(y).value_counts())
df = pd.DataFrame(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)  # startify = y 비율에 맞게 split 진행

print(np.unique(y_train, return_counts=True))  # (array([0, 1]), array([148, 250]))
print(np.unique(y_test, return_counts=True))  # (array([0, 1]), array([ 64, 107]))

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)

# scaler = MinMaxScaler().fit(x_train)
# scaler = StandardScaler().fit(x_train)
# scaler = MaxAbsScaler().fit(x_train)
scaler = RobustScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


PATH = './_save/keras33/'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = f'{PATH}k31_{date}-{filename}'

# 2. 모델 구성
# model = Sequential()
# model.add(Dense(30, input_dim=30, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


input = Input(shape=(30,))
dense1 = Dense(30, activation='relu')(input)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(40, activation='relu')(dense2)
dense4 = Dense(40, activation='relu')(dense3)
dense5 = Dense(40, activation='relu')(dense4)
dense6 = Dense(40, activation='relu')(dense5)
dense7 = Dense(40, activation='relu')(dense6)
output = Dense(1, activation='sigmoid')(dense7)
model = Model(inputs=input, outputs=output)

# model.summary()

# exit()

# 3. 컴파일, 훈련
learning_rate = 0.00001

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
    #   metrics=['accuracy'],
    metrics=['acc'],
)  # metrics = 주요 보조지표
es = EarlyStopping(monitor='val_loss', mode='min', patience=250, restore_best_weights=True)
mcp = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
rlr = ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)
start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=5000, batch_size=32, validation_split=0.3, callbacks=[es, rlr]
)
end_time = time.time()
print(start_time - end_time)

# 4. 예측, 평가
loss = model.evaluate(x_test, y_test)
print('=======================================')
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)  # 분류모델에서 sklearn metrics 사용을 위한 후처리
print(f'소요시간 = {end_time - start_time:.4f}')
print(f'learning_rate = {learning_rate}')
print(f'loss = {loss[0]:.4f}')
print(f'acc = {loss[1]:.4f}')

acc = accuracy_score(y_test, y_pred)  # accuracy 교차검증
print(f'sklearn acc = {acc:.4f}')
print('=======================================')

# CPU vs GPU
# GPU : 소요시간 = 21.5011
# CPU : 소요시간 = 32.5078


"""
MinMaxScaler 적용
loss = 0.0501
acc = 0.9883
sklearn acc = 0.9883

StandardScaler
loss = 0.0935
acc = 0.9591
sklearn acc = 0.9591

MaxAbsScaler
loss = 0.0609
acc = 0.9883
sklearn acc = 0.9883

RobustScaler
loss = 0.1187
acc = 0.9532
sklearn acc = 0.9532

Save
loss = 0.1425
acc = 0.9591
sklearn acc = 0.9591

learning_rate = 1e-05
loss = 0.1340
acc = 0.9474
sklearn acc = 0.9474

learning_rate = 1e-05
loss = 0.0975
acc = 0.9708
sklearn acc = 0.9708
"""

# # 5. 시각화
# plt.plot(hist.history['loss'], label='loss')
# plt.plot(hist.history['val_loss'], label='val_loss')
# plt.legend(loc='upper right')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('유방암 데이터')
# plt.grid()
# plt.show()
