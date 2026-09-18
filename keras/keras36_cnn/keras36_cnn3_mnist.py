import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential

#! 1. 데이터 -----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)
print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

#! 이미지 전처리

#! 스케일링 1 Minmax -----------------------------------------------------------
# * 스케일링은 x_train이 대상이다
# * 최소값이 0이기 떄문에 255로 나누는 거임 0~1
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# print(np.max(x_train), np.min(x_train))
# print(np.max(x_test), np.min(x_test))


#! 스케일링 2 Maxabs -----------------------------------------------------------
# # * -1.0 ~ 1.0
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

print(np.unique)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)

#! 원핫 인코더 -----------------------------------------------------------
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)  # ? (60000, 10) (10000, 10)

#! 2. 모델 구성 -----------------------------------------------------------
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3)))  # ? (26,26,64)

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))  # ? (24,24,32)
model.add(Dropout(0.2))

model.add(Conv2D(32, (2, 2), activation='relu'))  # ? (23, 23, 32)
model.add(Conv2D(16, (2, 2), activation='relu'))  # ? (22, 22, 16)
model.add(Dropout(0.2))

model.add(Conv2D(16, (2, 2), activation='relu'))  # ? (21, 21, 16)
model.add(Dropout(0.2))
model.add(Conv2D(8, (2, 2), activation='relu'))  # ? (21, 21, 16)
model.add(Dropout(0.2))
model.add(Conv2D(4, (2, 2), activation='relu'))  # ? (21, 21, 16)
model.add(Dropout(0.2))

model.add(Conv2D(2, (2, 2), activation='relu'))  # ? (20, 20, 16)

#! Flatten -----------------------------------------------------------
model.add(Flatten())
# model.add(Dense(units=32, activation='relu'))  # unit = 아웃풋 노드의 개수
model.add(Dropout(0.2))
# model.add(Dense(16, input_shape=(32,), activation='relu'))
model.add(Dense(10, activation='softmax'))  # ? (10,)

model.summary()

#! 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
)
end_time = time.time()

#! 4. 평가, 예측
print('---------------------------- model.evalute -------------------------------')
loss = model.evaluate(x_test, y_test, verbose=1)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)
y_test = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

# -----------------------------------------------------------

print('acc: ', loss[1])
print('loss: ', loss[0])
print('acc: ', acc)
print(round(end_time - start_time, 2), '초')

# 0.995 맞추기
"""
-----------------------------------------------------------
acc:  0.9898999929428101
loss:  0.04340008273720741
acc:  0.0987
188.33 초

"""
