import datetime
import time

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.1,
    fill_mode='nearest'
)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)

augment_size = 30000
# idx = np.random.randint(x_train.shape[0], size=augment_size) # randint 는 중복값이 가능. 원칙적으로는 choice를 사용해야하나 여기서는 중복이여도 증감도 각자 랜덤적용이라 사실상 문제는 없음
idx = np.random.choice(x_train.shape[0], size=augment_size)


x_aug = x_train[idx].copy()
y_aug = y_train[idx].copy()

x_aug = datagen.flow(x_aug, y_aug,
                     batch_size= augment_size,
                     shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_aug))
y_train = np.concatenate((y_train, y_aug))

print(x_train.shape, y_train.shape)

x_train = (x_train-127.5) / 127.5
x_test = (x_test-127.5) / 127.5

y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train)
y_test = OneHotEncoder(sparse_output=False).fit_transform(y_test)

print(y_train.shape)


# 2. 모델 구성

input = Input(shape=(32, 32, 3))
conv1 = Conv2D(32, (5,5), activation='relu')(input)
pool1 = MaxPooling2D()(conv1)
drop1 = Dropout(0.4)(pool1)
conv2 = Conv2D(64, (3,3), activation='relu')(drop1)
pool2 = MaxPooling2D()(conv2)
conv3 = Conv2D(64, (3,3), activation='relu')(pool2)
drop2 = Dropout(0.2)(conv3)
conv4 = Conv2D(32, (3,3), activation='relu')(drop2)
avgpool = GlobalAveragePooling2D()(conv4)
dense1 = Dense(64, activation='relu')(avgpool)
output = Dense(100, activation='softmax')(dense1)
model = Model(inputs=input, outputs=output)

model.summary()

# model = Sequential()
# model.add(Conv2D(32, (5,5), input_shape = (32, 32, 3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.4))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(GlobalAveragePooling2D())
# # model.add(Dense(256, activation='relu'))
# model.add(Dense(64, activation='relu'))
# # model.add(Dense(16, activation='relu'))
# model.add(Dense(100, activation='softmax'))

# model.summary()

# exit()

# 3. 컴파일, 훈련
learning_rate = 0.0005
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(patience=20, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=300,
          batch_size = 128,
          validation_split = 0.2,
          callbacks=[es, rlr],
          )
end_time = time.time()


# 4. 예측, 평가
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)
acc = accuracy_score(y_test, y_pred)


print(f'소요시간 = {end_time - start_time:.4f}')
print(f'learning_rate = {learning_rate}')
print(f'loss = {loss[0]:.4f}')
print(f'acc = {loss[1]:.4f}')
print(f'acc_score = {acc:.4f}')

# Epoch 20/100
# 313/313 [==============================] - 3s 10ms/step - loss: 1.4119 - acc: 0.5960 - val_loss: 3.7249 - val_acc: 0.2769
# Epoch 21/100
# 313/313 [==============================] - 3s 10ms/step - loss: 1.3439 - acc: 0.6135 - val_loss: 3.8288 - val_acc: 0.2774
# 313/313 [==============================] - 1s 2ms/step - loss: 2.8853 - acc: 0.3047
# 313/313 [==============================] - 0s 1ms/step
# 소요시간 = 71.7929
# loss = 2.8853
# acc = 0.3047
# acc_score = 0.3047

# 소요시간 = 57.4047
# loss = 2.4505
# acc = 0.3786
# acc_score = 0.3786

# 소요시간 = 57.5599
# loss = 2.3926
# acc = 0.3853
# acc_score = 0.3853

# 소요시간 = 128.2734
# loss = 2.3051
# acc = 0.4135
# acc_score = 0.4135

# 소요시간 = 148.5203
# loss = 2.2239
# acc = 0.4275
# acc_score = 0.4275

# 소요시간 = 75.6433
# loss = 3.5446
# acc = 0.1858
# acc_score = 0.1858

### aug size = 50000 ###
# 소요시간 = 356.3497
# loss = 2.1014
# acc = 0.4496
# acc_score = 0.4496

# 소요시간 = 257.3357
# learning_rate = 0.0005
# loss = 2.0702
# acc = 0.4564
# acc_score = 0.4564

# 소요시간 = 206.1716
# learning_rate = 0.0005
# loss = 2.0517
# acc = 0.4627
# acc_score = 0.4627