import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ======================================================================
SEED = 333

# --- 데이터 ---
PATH = './_save/keras46/rps/'
VAL_SPLIT = 0.15

# --- 모델 ---
TRAINABLE = True  # True: 동결 안 함(3번) / False: 가중치 동결(2번)
DENSE_UNITS = 64

# --- 훈련 ---
EPOCHS = 100
PATIENCE = 10  # EarlyStopping
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SUBJECT = 'rps_vgg16'
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#! 1. 데이터 ------------------------------------------------------
# keras46_save_npy_02_rps.py 에서 저장해 둔 npy 를 그대로 불러온다 (rescale 1/255, 원핫 적용 끝)
x_train = np.load(PATH + 'keras46_rps_x_train.npy')
x_test = np.load(PATH + 'keras46_rps_x_test.npy')
y_train = np.load(PATH + 'keras46_rps_y_train.npy')
y_test = np.load(PATH + 'keras46_rps_y_test.npy')
print(x_train.shape, y_train.shape)  # ? (1536, 200, 200, 3) (1536, 3)
print(x_test.shape, y_test.shape)  # ? (512, 200, 200, 3) (512, 3)

INPUT_SHAPE = x_train.shape[1:]
OUTPUT_DIM = y_train.shape[1]  # 원핫이라 클래스 수 = 3

#! 2. model ------------------------------------------------------
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=INPUT_SHAPE,
)
vgg16.trainable = TRAINABLE

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(DENSE_UNITS))
model.add(Dense(DENSE_UNITS))
model.add(Dense(DENSE_UNITS))
model.add(Dense(OUTPUT_DIM, activation='softmax'))

model.summary()

#! 3. compile & fit -------------------------------------------------------------
model.compile(
    loss='categorical_crossentropy',
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
acc = accuracy_score(np.argmax(y_test, axis=-1), y_predict)  # y_test 는 원핫이라 argmax

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

"""
======================= 실험 기록 =======================
| subject=rps_vgg16 | seed=333 | trainable=False | dense=64 | vs=0.15 | bs=32 | ep=100 | lr=0.001 | pat=10
|| loss=0.0000 | acc=1.0000 | stop=ES미발동 | time=451.2s | 4511.5ms/epoch |

| subject=rps_vgg16 | seed=333 | trainable=True | dense=64 | vs=0.15 | bs=32 | ep=100 | lr=0.001 | pat=10
|| loss=0.0000 | acc=1.0000 | stop=71 | time=841.9s | 11857.4ms/epoch |
"""
