# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
import datetime
import time

import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam

PATH = './_data/kaggle_bike/'

# 1. 데이터
train = pd.read_csv(PATH + 'train.csv', index_col=0)
test = pd.read_csv(PATH + 'test.csv', index_col=0)

print(train.shape, test.shape)   # (10886, 11) (6493, 8)
print(train.info(), test.info()) 

x = train.drop(columns=['casual', 'registered', 'count'])
y = train['count']
print(x.shape, test.shape)   # (10886, 8) (6493, 8)
print(x.info(), test.info()) 

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)

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
# model.add(Dense(10, input_dim=8, activation='relu'))
# model.add(Dense(10, activation='relu'))
# # model.add(Dropout(0.3))
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
output = Dense(1)(dense5)
model = Model(inputs=input, outputs=output)

# model.summary()
# exit()

# 3. 컴파일, 훈련
learning_rate = 0.0005
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(patience=350, restore_best_weights=True)
# mcp = ModelCheckpoint(filepath=filepath, verbose=1,
#                       save_best_only=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=256, validation_split=0.2, callbacks=[es])
end_time = time.time()


# 4. 예측, 평가
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print(f'소요시간 = {end_time - start_time:.4f}')
print(f'learning_rate = {learning_rate}')
print(f'r2 = {r2:.4f}')
print(f'rmse = {rmse:.4f}')

# CPU vs GPU
# GPU : 소요시간 = 29.9990
# CPU : 소요시간 = 38.2308

# epochs=10000, batch_size=32, validation_split=0.3
# patience = 100
# r2 = 0.2690
# rmse = 154.8744

"""
MinMaxscaler 
r2 = 0.2698
rmse = 154.7886

StandardScaler
r2 = 0.2693
rmse = 154.8481

MaxAbsScaler
r2 = 0.2694
rmse = 154.8340

RobustScaler
r2 = 0.2695
rmse = 154.8236

Save
r2 = 0.2695
rmse = 154.8264

with activation
r2 = 0.3273
rmse = 148.5706

with Dropout
성능저하

learning_rate = 0.01
r2 = 0.3339
rmse = 147.8375

learning_rate = 0.0005
r2 = 0.3256
rmse = 148.7635
"""


# # 5. 시각화
# plt.plot(hist.history['loss'], label='loss')
# plt.plot(hist.history['val_loss'], label='val_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.title('kaggle_earlystopping')
# plt.show()