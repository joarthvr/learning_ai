import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, MaxPool2D
from tensorflow.keras.models import Model

SEED = 42
EPOCHS = 50

BATCH_SIZE = 128
VAL_SPLIT = 0.2
STEP = 'fashion'
PATIENCE = 15
RLR_PATIENCE = 5

#! 1. 데이터 -----------------------------------------------------------

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  # (10000, 32, 32, 3) (10000, 1)
print(np.max(x_train), np.min(x_train))  # 255 0 => 흑백
print(np.max(x_test), np.min(x_test))  # 255 0 => 흑백

#! 이미지 전처리
# -1 ~ 1
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

#! 원핫 인코더
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

#! 2. 모델 구성 -----------------------------------------------------------


inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x = MaxPool2D()(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D()(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPool2D()(x)

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(100, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

#! 3. 컴파일, 훈련 -----------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=PATIENCE,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=VAL_SPLIT,
    callbacks=[es],
)
end_time = time.time()

#! 4. 평가, 예측 -----------------------------------------------------------
loss = model.evaluate(x_test, y_test, verbose=0)

y_predict = np.argmax(model.predict(x_test), axis=1)
y_test_label = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_label, y_predict)

print('')
print('===== RESULT =====')
print(
    f'|| acc={acc:.4f} | loss={loss:.4f} '
    f'| stop={es.stopped_epoch if es.stopped_epoch else "ES미발동"} '
    f'| time={end_time - start_time:.1f}s |'
)
print('===================================')
print('')

"""
======================= 실험 기록 =======================
# 0.4 목표
===== RESULT =====
? | step=fashion | seed=42 | bs=128 | ep=50 | pat=15
? || acc=0.4667 | loss=2.1094 | stop=ES미발동 | time=138.6s |

| step=fashion | seed=42 | bs=128 | ep=50 | pat=15
|| acc=0.4544 | loss=2.3073 | stop=49 | time=411.2s |
===================================

"""
