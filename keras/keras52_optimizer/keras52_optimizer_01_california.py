# 35-1 복사

import time

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False

path = './_save/keras30/'

# 1. 데이터
housing = fetch_california_housing()
x = housing.data
y = housing.target

print(x.shape, y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


# 2. 모델 구성
# model = Sequential()
# model.add(Dense(10, input_dim=8, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

# model.summary()


input = Input(shape=(8,))
dense1 = Dense(10, activation='relu')(input)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output = Dense(1, activation='relu')(dense5)
model = Model(inputs=input, outputs=output)


# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001  # Default
# learning_rate = 0.0003
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.01


model.compile(
    loss='mse', optimizer=Adam(learning_rate=learning_rate)
)  # weights만 불러와서 compile도 해야함
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)
# ################# mcp 세이브 파일명 만들기 시작 ####################
# import datetime
# date = datetime.datetime.now()
# print(date)     # 2026-09-14 11:44:56.023886
# print(type(date))   # <class ' datetime.datetime')
# date = date.strftime("%m%d_%H%M")
# print(date)   # 0914_1148
# print(type(date))   # <class 'str'>

# path = './_save/keras30/'
# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = ''.join([path, 'k30_', date, '-', filename])

############# 이런식으로 파일명 관리하면 편할 것임 ################

# 내가 생각하는 파일명 예.
# ./_save/keras30/ + k30_ + 0914_1148 + '530-0.001.keras'

# exit()


# mcp = ModelCheckpoint(
#     monitor = 'val_loss',
#     mode = 'auto',
#     save_best_only = True,
#     filepath = filepath,
#     verbose=1
# )

start_time = time.time()
hist = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es])
end_time = time.time()


# 4. 평가, 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print(f'소요시간 = {end_time - start_time:.4f}')
print(f'learning_rate = {learning_rate}')
print(f'r2 = {r2:.4f}')
print(f'RMSE = {rmse:.4f}')

# CPU vs GPU

# GPU : 소요시간 = 59.0145
# CPU : 소요시간 = 36.6595


# validation_split=0.2, epochs=1000, batch_size=512
# r2 = 0.4409
# RMSE = 0.8485

# MinMaxScaler 전처리 후
# r2 = 0.5981
# RMSE = 0.7192

# MinMaxScaler 제대로 된 전처리 후 (x_train 기준 scaler.fit())
# r2 = 0.5971
# RMSE = 0.7355

# StandardScaler
# r2 = 0.5942
# RMSE = 0.7217

# MaxAbsScaler
# r2 = 0.5580
# RMSE = 0.7614

# RobustScaler
# r2 = 0.6134
# RMSE = 0.7113

# RobustScaler 2 with es & mcp
# r2 = 0.6352
# RMSE = 0.6917

# RobustScaler with 2 with es & mcp & activation
# r2 = 0.7929
# RMSE = 0.5278

# learning_rate = 0.01
# 소요시간 = 57.7927
# r2 = 0.7660
# RMSE = 0.5558

# learning_rate = 0.003
# 소요시간 = 57.7927
# r2 = 0.7660
# RMSE = 0.5558
