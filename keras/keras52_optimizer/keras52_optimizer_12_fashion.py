# 50-2 복사

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

################### 증폭 ###################
datagen = ImageDataGenerator(           # 이 중에 랜덤으로 몇개 적용
    horizontal_flip=True, # 수평 뒤집기 (좌우반전)
    # vertical_flip=True, # 수집 뒤집기 (상하반전)
    width_shift_range=0.1, # 평행 이동
    # height_shift_range=0.1,
    rotation_range=15,   # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.1, 
    # shear_range=0.7, # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode = 'nearest', 
)

augment_size = 40000
print(x_train.shape)
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 6만개중에 4만개 랜덤뽑기
print(randidx.shape) # .shape는 array, vector 경우
print(len(randidx)) # list나 tuple은 len으로 확인해야함

print(np.min(randidx), np.max(randidx))

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],1)
print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28, 1) (40000,)

x_augmented = datagen.flow(
                x_augmented, y_augmented,
                batch_size=augment_size,
                shuffle=False
).next()[0]

####### 변환 완료 #######
print(x_augmented.shape)    # (40000, 28, 28, 1)


print(x_train.shape) # (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))/255
x_test = x_test/255
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

print(np.unique(y_train, return_counts=True)) # 데이터 불균형 확인 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([ 9979, 10015,  9876, 10005, 10060, 10039,  9996,  9999, 10081,
#         9950], dtype=int64))

y_train = pd.get_dummies(y_train, dtype=int)
print(y_train.shape) # (100000, 10)
y_test = pd.get_dummies(y_test, dtype=int)
print(y_test.shape)    # (10000, 10)

# 2. 모델 구성

input = Input(shape=(28, 28, 1))
conv1 = Conv2D(32, (3,3), activation='relu')(input)
pool = MaxPooling2D()(conv1)
# drop1 = Dropout(0.2)
conv2 = Conv2D(32, (3,3), activation='relu')(pool)
drop2 = Dropout(0.2)(conv2)
conv3 = Conv2D(64, (3,3), activation='relu')(drop2)
conv4 = Conv2D(64, (3,3), activation='relu')(conv3)
drop3 = Dropout(0.2)(conv4)
avgpool = GlobalAveragePooling2D()(drop3)
dense1 = Dense(16, activation='relu')(avgpool)
output = Dense(10, activation='softmax')(dense1)
model = Model(inputs=input, outputs=output)
model.summary()


# model = Sequential()
# model.add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation='relu'))
# model.add(MaxPooling2D())
# # model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

# exit()

# 3. 컴파일, 훈련
learning_rate = 0.005
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(patience=20, restore_best_weights=True)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=150,
          batch_size = 64,
          validation_split = 0.2,
          callbacks=[es],
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


# 소요시간 = 264.1396
# loss = 0.3027
# acc = 0.8940

# 소요시간 = 71.4311
# loss = 0.2959
# acc = 0.9052
# acc_score = 0.9052

# 소요시간 = 121.0216
# loss = 0.2230
# acc = 0.9239
# acc_score = 0.9239

# 소요시간 = 179.6226
# loss = 0.2328
# acc = 0.9183
# acc_score = 0.9183

# 소요시간 = 254.0470
# loss = 0.2299
# acc = 0.9254
# acc_score = 0.9254

# learning_rate = 0.0001
# loss = 0.2499
# acc = 0.9103
# acc_score = 0.9103

# learning_rate = 0.005
# loss = 0.2554
# acc = 0.9099
# acc_score = 0.9099