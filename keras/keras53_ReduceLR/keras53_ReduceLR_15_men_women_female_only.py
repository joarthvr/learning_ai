# 47-3 복사

import datetime
import time

import numpy as np
from keras.preprocessing.image import ImageDataGenerator  # tensorflow 빼도 됌 오히려 버전 2는 이게 편함
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam

# 0.95

# 1. 데이터

np_path = '_save/keras51/gender/'

x_train = np.load(np_path + 'keras51_gender_x_train.npy')
x_test = np.load(np_path + 'keras51_gender_x_test.npy')
y_train = np.load(np_path + 'keras51_gender_y_train.npy')
y_test = np.load(np_path + 'keras51_gender_y_test.npy')

print(x_train.shape, x_test.shape)  # (26375, 100, 100, 3) (8792, 100, 100, 3)
print(y_train.shape, y_test.shape)  # (26375,) (8792,)

print(np.unique(y_train, return_counts=True)) # (array([0., 1.], dtype=float32), array([13258, 13117], dtype=int64))

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = f'{np_path}k48_{date}-{filename}'

# 2. 모델 구성

input = Input(shape=(100,100,3))
conv1 = Conv2D(16, (5,5), activation='relu')(input)
conv2 = Conv2D(16, (5,5), activation='relu')(conv1)
pool1 = MaxPooling2D()(conv2)

conv3 = Conv2D(32, (3,3), activation='relu')(pool1)
conv4 = Conv2D(32, (3,3), activation='relu')(conv3)
pool2 = MaxPooling2D()(conv4)

conv5 = Conv2D(64, (3,3), activation='relu')(pool2)
conv6 = Conv2D(64, (3,3), activation='relu')(conv5)
pool3 = MaxPooling2D()(conv6)

avgpool = GlobalAveragePooling2D()(pool3)
dense1 = Dense(64, activation='relu')(avgpool)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(1, activation='sigmoid')(dense2)

model = Model(inputs = input, outputs=output)

model.summary()

# 3. 컴파일, 훈련
learning_rate = 0.0001
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(patience=20, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)
mcp = ModelCheckpoint(filepath=filepath,
                      save_best_only=True,
                      verbose=1)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=3000,
          batch_size=16,
          callbacks = [es, rlr],
          validation_split=0.3)
end_time = time.time()


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

print(f'소요시간 = {end_time - start_time:.4f}')
print(f'learning_rate = {learning_rate}')
print(f'loss = {loss[0]:.4f}')
print(f'acc = {loss[1]:.4f}')

# 소요시간 = 275.4344
# loss = 0.1993
# acc = 0.9201

# 소요시간 = 470.6086
# loss = 0.1944
# acc = 0.9227

# 소요시간 = 253.3460
# loss = 0.1915
# acc = 0.9245

# learning_rate = 0.0001
# loss = 0.2107
# acc = 0.9198