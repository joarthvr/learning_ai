# 40-1 복사

import time

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# 1. 데이터
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.1,
    fill_mode='nearest'
)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)  # (60000, 28, 28), (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

augment_size = 40000
idx = np.random.randint(x_train.shape[0], size=40000)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_aug = x_train[idx].copy()
y_aug = y_train[idx].copy()

print(x_aug.shape, y_aug.shape) # (40000, 28, 28) (40000,)

x_aug = datagen.flow(x_aug, y_aug,
             batch_size=augment_size,
             shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_aug))
y_train = np.concatenate((y_train, y_aug))

print(x_train.shape, y_train.shape) # 100000, 28, 28, 1) (100000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000,)

#### 스케일링 2
x_train = (x_train - 127.5) / 127.5  # MaxAbs 수동 스케일링 
x_test = (x_test - 127.5) / 127.5  
print(np.max(x_train), np.min(x_train))  # -1.0 0.0
print(np.max(x_test), np.min(x_test))    # -1.0 0.0

# 이미지의 경우 이런식으로도 전처리 많이 함. 애초에 범위가 한정되어 있기 때문

#### OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)

y_test = y_test.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) # (100000, 10) (10000, 10)

# 2. 모델 구성

input = Input(shape=(28,28,1))
conv1 = Conv2D(16, (2,2), activation='relu')(input)
pool1 = MaxPooling2D()(conv1)
conv2 = Conv2D(32, (3,3), activation='relu')(pool1)
drop1 = Dropout(0.2)(conv2)
conv3 = Conv2D(32, (3,3), activation='relu')(drop1)
conv4 = Conv2D(32, (3,3), activation='relu')(conv3)
drop2 = Dropout(0.2)(conv4)
conv5 = Conv2D(64, (3,3), activation='relu')(drop2)
drop3 = Dropout(0.2)(conv5)
conv6 = Conv2D(64, (3,3), activation='relu')(drop3)
avgpool = GlobalAveragePooling2D()(conv6)
dense1 = Dense(64, activation='relu')(avgpool)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(10, activation='softmax')(dense2)

model = Model(inputs=input, outputs=output)

model.summary()

# exit()

# model = Sequential()
# model.add(Conv2D(16, (2,2), input_shape=(28, 28, 1), activation='relu'))   # (26, 26, 64) / # linear라 activation 그대로 사용가능 
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) # (24, 24, 32)
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu')) # (23, 23, 32)  # 파라미터 어느 방식으로든 써도 상관 X
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) # (22, 22, 16)
# model.add(Dropout(0.2))
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu')) # (21, 21, 16)
# model.add(Dropout(0.2))
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu')) 
# model.add(GlobalAveragePooling2D()) # 20,20,16 을 Dense에 들어가게끔 펼쳐줌
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(10, activation='softmax'))  # (10,)  

# model.summary()

# exit()

# 3. 컴파일, 훈련
learning_rate = 0.0001
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
              metrics = ['acc'])
es = EarlyStopping(patience=20, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)
start_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=150,
          verbose=1,
          batch_size=128,
          validation_split=0.2,
          callbacks=[es,rlr]
          )
end_time = time.time()

# 4. 예측, 평가
print('==============model.evaluate===============')
loss = model.evaluate(x_test, y_test, verbose=1)
print(f'loss = {loss[0]:.4f}')
print(f'acc = {loss[1]:.4f}')

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print(f'learning_rate = {learning_rate}')
print(f'acc_score = {acc}')
print(f'걸린시간 = {end_time - start_time:.2f}')



# CPU
# loss = 0.03590594604611397
# acc = 0.9914000034332275
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
# acc_score = 0.9914
# 걸린시간 = 900.81

# GPU
# loss = 0.0519
# acc = 0.9897
# 313/313 [==============================] - 0s 1ms/step
# acc_score = 0.9897
# 걸린시간 = 120.30


# 0.995 맞추기

# loss = 0.0241
# acc = 0.9944
# 313/313 [==============================] - 0s 951us/step
# acc_score = 0.9944
# 걸린시간 = 79.12


# loss = 0.0168
# acc = 0.9955
# 313/313 [==============================] - 0s 1ms/step
# acc_score = 0.9955
# 걸린시간 = 54.99


# loss = 0.0197
# acc = 0.9950
# 313/313 [==============================] - 0s 1ms/step
# acc_score = 0.995
# 걸린시간 = 136.06

# loss = 0.0895
# acc = 0.9788
# 313/313 [==============================] - 0s 912us/step
# learning_rate = 0.01
# acc_score = 0.9788
# 걸린시간 = 102.02

# loss = 0.0214
# acc = 0.9939
# 313/313 [==============================] - 0s 1ms/step
# learning_rate = 0.0001
# acc_score = 0.9939
# 걸린시간 = 286.70

# loss = 0.0218
# acc = 0.9933
# 313/313 [==============================] - 0s 979us/step
# learning_rate = 0.0001
# acc_score = 0.9933
# 걸린시간 = 195.87