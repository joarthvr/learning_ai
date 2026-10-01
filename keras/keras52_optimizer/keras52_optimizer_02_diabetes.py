import time

from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# PATH = './_save/keras33/'

# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")
# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = f'{PATH}k33_{date}-{filename}'


# 1. 데이터
dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


# 2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim = 10, activation='relu',))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

# model.summary()

input = Input(shape=(10,))
dense1 = Dense(10, activation='relu')(input)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(10, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(10, activation='relu')(drop2)
dense4 = Dense(10, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output = Dense(1, activation='relu')(dense5)
model = Model(inputs=input, outputs=output)

# model.summary()

# exit()
# 3. 컴파일, 훈련
learning_rate = 0.0001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
es = EarlyStopping(patience=100, restore_best_weights=True)
# mcp = ModelCheckpoint(verbose=1,
#                       save_best_only=True,
#                       filepath = filepath)

start_time = time.time()
model.fit(x_train, y_train, epochs=5000, batch_size=32, callbacks=[es], validation_split=0.2)
end_time = time.time()


# 4. 예측, 평가
y_pred = model.predict(x_test)
result = model.evaluate(x_test, y_test)
print(f'소요시간 : {end_time - start_time:.4f}')
print(f'learning_rate = {learning_rate}')
print(f'result = {result:.4f}')
print(f'r2 = {r2_score(y_test, y_pred):.4f}')
print(f'rmse = {root_mean_squared_error(y_test, y_pred):.4f}')

# CPU vs GPU
# GPU : 소요시간 : 17.2245
# CPU : 소요시간 : 29.9512

# result = 3022.2603
# r2 = 0.4534
# rmse = 54.9751

# result = 3022.2603
# r2 = 0.4534
# rmse = 54.9751

# r2 = 0.4739
# rmse = 53.9358

# learning_rate = 0.0001
# 소요시간 : 91.7342
# result = 2918.2776
# r2 = 0.4723
# rmse = 54.0211
