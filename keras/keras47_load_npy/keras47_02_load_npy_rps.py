import time

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model

# 1. 데이터

np_path = '_save/keras46/rps/'

x_train = np.load(np_path + 'keras46_rps_x_train.npy')
y_train = np.load(np_path + 'keras46_rps_y_train.npy')
x_test = np.load(np_path + 'keras46_rps_x_test.npy')
y_test = np.load(np_path + 'keras46_rps_y_test.npy')

print(x_train.shape, x_test.shape)  # (1536, 200, 200, 3) (512, 200, 200, 3)
print(y_train.shape, y_test.shape)  # (1536, 3) (512, 3)

# 2. 모델 구성

input = Input(shape=(200, 200, 3))
conv1 = Conv2D(16, (5, 5), activation='relu')(input)
conv2 = Conv2D(16, (5, 5), activation='relu')(conv1)
pool1 = MaxPooling2D()(conv2)

conv3 = Conv2D(32, (5, 5), activation='relu')(pool1)
conv4 = Conv2D(32, (5, 5), activation='relu')(conv3)
pool2 = MaxPooling2D()(conv4)

conv5 = Conv2D(64, (5, 5), activation='relu')(pool2)
conv6 = Conv2D(64, (5, 5), activation='relu')(conv5)
drop2 = Dropout(0.2)(conv6)

avgpool = GlobalAveragePooling2D()(drop2)
dense1 = Dense(64, activation='relu')(avgpool)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(3, activation='softmax')(dense2)

model = Model(inputs=input, outputs=output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(patience=15, restore_best_weights=True)
start_time = time.time()
model.fit(x_train, y_train, epochs=70, batch_size=128, callbacks=[es], validation_split=0.3)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print(f'loss = {loss[0]:.4f}')
print(f'acc = {loss[1]}')
print(f'소요시간 = {end_time - start_time:.4f}')

# loss = 0.0008
# acc = 1.0
# 소요시간 = 123.8181
