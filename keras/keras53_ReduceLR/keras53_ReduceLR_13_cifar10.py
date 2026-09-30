import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 0. 증강 준비
datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    rotation_range=20
)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(np.unique(y_train, return_counts=True))   # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

augment_size = 30000
idx = np.random.randint(x_train.shape[0], size=augment_size)

x_aug = x_train[idx].copy()
y_aug = y_train[idx].copy()

x_aug = datagen.flow(x_aug, y_aug,
                     batch_size=augment_size,
                     shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_aug))
y_train = np.concatenate((y_train, y_aug))


y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train)
y_test = OneHotEncoder(sparse_output=False).fit_transform(y_test)

#### 스케일링
x_train = (x_train - 127.5) / 127.5  # MaxAbs 수동 스케일링 
x_test = (x_test - 127.5) / 127.5  
print(np.max(x_train), np.min(x_train))  # -1.0 0.0
print(np.max(x_test), np.min(x_test))    # -1.0 0.0


print(x_train.shape, y_train.shape) # 100000, 28, 28, 1) (100000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000,)

# 2. 모델 구성
input = Input(shape=(32, 32, 3))
conv1 = Conv2D(32, (3,3), input_shape = (32, 32, 3), activation='relu')(input)
pool = MaxPooling2D()(conv1)
conv2 = Conv2D(32, (3,3), activation='relu')(pool)
drop1 = Dropout(0.4)(conv2)
conv3 = Conv2D(64, (3,3), activation='relu')(drop1)
conv4 = Conv2D(64, (3,3), activation='relu')(conv3)
drop2 = Dropout(0.3)(conv4)
conv5 = Conv2D(16, (3,3), activation='relu')(drop2)
avgpool = GlobalAveragePooling2D()(conv5)
dense1 = Dense(10, activation='relu')(avgpool)
output = Dense(10, activation='softmax')(dense1)
model = Model(inputs=input, outputs=output)

model.summary()

# model = Sequential()
# model.add(Conv2D(32, (3,3), input_shape = (32, 32, 3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(Dropout(0.4))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (3,3), activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

# 3. 컴파일, 훈련
learning_rate = 0.0005
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(patience=20, restore_best_weights=True)
rlr = ReduceLROnPlateau(patience=20, factor=0.5, verbose=1)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=150,
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

# 소요시간 = 132.5162
# loss = 1.4195
# acc = 0.6720
# acc_score = 0.6720

# 소요시간 = 83.3630
# loss = 0.7334
# acc = 0.7604
# acc_score = 0.7604

# 소요시간 = 96.8098
# loss = 0.7088
# acc = 0.7641
# acc_score = 0.7641

# 소요시간 = 35.7301
# loss = 1.7474
# acc = 0.5416
# acc_score = 0.5416

### aug size = 50000 ###
# 소요시간 = 289.8035
# loss = 0.5911
# acc = 0.8035
# acc_score = 0.8035

### aug size = 30000 ###
# 소요시간 = 225.4219
# loss = 0.5917
# acc = 0.8090
# acc_score = 0.8090

# learning_rate = 0.0005
# loss = 0.6040
# acc = 0.8036
# acc_score = 0.8036

# 소요시간 = 280.4340
# learning_rate = 0.0005
# loss = 0.5974
# acc = 0.8103
# acc_score = 0.8103