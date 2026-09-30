import datetime
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

# acc = 0.95

# 1. 데이터
dataset = load_wine(as_frame=True)
x = dataset.data
y = pd.DataFrame(dataset.target)

print(x.shape, y.shape)     # (178, 13) (178,)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48]))
# print(dataset.DESCR)        # 13 columns, 3 classes
# print(dataset.feature_names)

enc = OneHotEncoder(sparse_output=False)
y = enc.fit_transform(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3, stratify=y)

# scaler = MinMaxScaler().fit(x_train)
# scaler = StandardScaler().fit(x_train)
# scaler = MaxAbsScaler().fit(x_train)
scaler = RobustScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


PATH = "./_save/keras31/"
filename = '{epoch:04d}-{val_loss:.4f}.keras'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = f'{PATH}k31_{date}-{filename}'


# 2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
learning_rate = 0.01
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(patience=300, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=10, factor=0.5, verbose=1)
# mcp = ModelCheckpoint(filepath=filepath,
#                       verbose=1,
#                       save_best_only=True)
start = time.time()
hist = model.fit(x_train, y_train,
          epochs = 5000,
          batch_size = 32,
          callbacks=[es, rlr],
          validation_split = 0.2)
end = time.time()
# model = load_model(f'{PATH}k31_0914_1438-0032-0.1310.keras')

# 4. 예측, 평가
result = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print(f'소요시간 = {end - start:.4f}')
print(f'learning_rate = {learning_rate}')
print(f'loss = {result[0]:.4f}')
print(f'acc = {result[1]:.4f}')


# CPU vs GPU
# GPU : 소요시간 = 12.5798
# CPU : 소요시간 = 28.9900


"""
epochs = 10000,
batch_size = 32,
callbacks=[es],
validation_split = 0.2

patience=320

loss = 0.1921
acc = 0.9259


MinMaxScaler 적용 
loss = 0.2279
acc = 0.9444

StandardScaler 적용
loss = 0.0150
acc = 1.0000

MaxAbsScaler 적용
소요시간 = 21.2439
loss = 0.0388
acc = 0.9815

RobustScaler
소요시간 = 21.6601
loss = 0.0773
acc = 0.9815

Save
소요시간 = 22.1965
loss = 0.1467
acc = 0.9630

load
loss = 0.1467
acc = 0.9630

learning_rate = 0.01
loss = 0.0000
acc = 1.0000

learning_rate = 0.01 with rlr
loss = 0.2658
acc = 0.9259
"""