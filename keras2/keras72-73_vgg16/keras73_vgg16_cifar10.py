import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ======================================================================
SEED = 333

# --- 데이터 ---
VAL_SPLIT = 0.15

# --- 모델 ---
TRAINABLE = False  # True: 동결 안 함(3번) / False: 가중치 동결(2번)
DENSE_UNITS = 64

# --- 훈련 ---
EPOCHS = 30
PATIENCE = 5  # EarlyStopping
BATCH_SIZE = 256
LEARNING_RATE = 0.001

SUBJECT = 'cifar10_vgg16'
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#! 1. 데이터 ------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  # ? (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  # ? (10000, 32, 32, 3) (10000, 1)

# 픽셀은 0~255 고정 상수라 train 기준으로 fit 할 필요가 없다 -> 데이터 누수 아님
x_train = x_train / 255.0
x_test = x_test / 255.0

# y 는 원핫하지 않고 숫자 라벨 그대로 쓴다 -> sparse_categorical_crossentropy
y_train = y_train.ravel()  # (N, 1) -> (N,)
y_test = y_test.ravel()
print(x_train.shape, y_train.shape)  # ? (50000, 32, 32, 3) (50000,)
print(x_test.shape, y_test.shape)  # ? (10000, 32, 32, 3) (10000,)

#! 2. model ------------------------------------------------------
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)
vgg16.trainable = TRAINABLE

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(DENSE_UNITS))
model.add(Dense(DENSE_UNITS))
model.add(Dense(DENSE_UNITS))
model.add(Dense(10, activation='softmax'))

model.summary()

#! 3. compile & fit -------------------------------------------------------------
# val_loss 가 PATIENCE 에폭 동안 좋아지지 않으면 멈춘다
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics=['acc'],
)

es = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
)

start_time = time.time()
hist = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=VAL_SPLIT,
    callbacks=[es],
)
end_time = time.time()

#! 4. evaluate, predict -------------------------------------------------------------
loss, _ = model.evaluate(x_test, y_test, verbose=0)
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 -> 라벨
acc = accuracy_score(y_test, y_predict)  # y_test 는 이미 숫자 라벨이다

##################### 실험 결과 요약 (기록용) #####################
took = end_time - start_time

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT} | seed={SEED} '
    f'| trainable={TRAINABLE} | dense={DENSE_UNITS} '
    f'| vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| loss={loss:.4f} | acc={acc:.4f} '
    f'| stop={es.stopped_epoch if es.stopped_epoch else "ES미발동"} '
    f'| time={took:.1f}s | {took / (es.stopped_epoch or EPOCHS) * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')

# 실습
# 비교할 거
# 1 이전에 본인이 한 최상의 결과
# 2 가중치를 동결하고 훈련시켰을 떄, trainable=False
# 3 가중치를 동결하지 않고 훈련시켰을 때, trainable=True
# 4 시간까지 비교

"""
======================= 실험 기록 =======================
| subject=cifar10_vgg16 | seed=333 | trainable=True | dense=100 | vs=0.15 | bs=256 | ep=30 | lr=0.001 | pat=5
|| loss=2.3029 | acc=0.1000 | stop=6 | time=86.4s | 14394.1ms/epoch |

| subject=cifar10_vgg16 | seed=333 | trainable=False | dense=64 | vs=0.15 | bs=256 | ep=30 | lr=0.001 | pat=5
|| loss=1.2077 | acc=0.5797 | stop=11 | time=56.8s | 5166.8ms/epoch |
"""
