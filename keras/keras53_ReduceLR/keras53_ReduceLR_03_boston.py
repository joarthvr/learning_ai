import datetime
import time

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, y_train.shape)   # (404, 13) (404,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler().fit(x_train)
scaler = RobustScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
# model = Sequential()
# model.add(Dense(10, input_dim=13, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(1))

# model.summary()

input = Input(shape=(13,))
dense1 = Dense(10, activation='relu')(input)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
# drop1 = Dropout(0.2)(dense4)
output = Dense(1)(dense4)
model = Model(inputs=input, outputs=output)

# model.summary()
# exit()

PATH = "./_save/keras33/"
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = f'{PATH}k31_{date}-{filename}'


# 3. 컴파일, 훈련
learning_rate = 0.01
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(patience=300)
rlr = ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)
# mcp = ModelCheckpoint(filepath = filepath,
#                       verbose=1,
#                       save_best_only=True,
#                       )
start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs = 5000,
                 batch_size=32,
                 callbacks=[es, rlr],
                 validation_split=0.3)
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
# GPU : 소요시간 = 15.9013
# CPU : 소요시간 = 30.1258


# epochs=10000, batch_size=32, validation_split=0.3
# patience = 100
# r2 = 0.7316
# rmse = 4.7271


"""
MinMaxScaler 전처리 후

r2 = 0.7502
rmse = 4.5598

StandardScaler 
r2 = 0.7454
rmse = 4.6034

MaxAbsScaler
r2 = 0.7508
rmse = 4.5544

RobustScaler
r2 = 0.7466
rmse = 4.5931

Save & load
r2 = 0.7453
rmse = 4.6043

with activation
r2 = 0.7850
rmse = 4.2309

with Dropout  (dropout 을 하면 빠짐)
r2 = 0.7636
rmse = 4.4361

with rlr
r2 = 0.8099
rmse = 3.9781
"""

# # 5. 시각화
# plt.plot(hist.history['loss'], label='loss')
# plt.plot(hist.history['val_loss'], label='val_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('boston_EarlyStopping')
# plt.legend()
# plt.grid()
# plt.show()