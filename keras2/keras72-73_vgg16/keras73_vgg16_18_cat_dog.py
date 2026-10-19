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
PATH = './_data/kaggle_cat_dog_npy/'
PREFIX = 'keras45_03_'  # keras45_03_catdog_save_npy.py 의 save 쪽과 같아야 한다
VAL_SPLIT = 0.15

# --- 모델 ---
TRAINABLE = False  # True: 동결 안 함(3번) / False: 가중치 동결(2번)
DENSE_UNITS = 64

# --- 훈련 ---
EPOCHS = 100
PATIENCE = 10  # EarlyStopping
BATCH_SIZE = 64
LEARNING_RATE = 0.001

SUBJECT = 'cat_dog_vgg16'
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#! 1. 데이터 ------------------------------------------------------
# keras45_03 에서 저장해 둔 npy 를 그대로 불러온다
x_train = np.load(PATH + PREFIX + 'x_train.npy')
x_test = np.load(PATH + PREFIX + 'x_test.npy')
y_train = np.load(PATH + PREFIX + 'y_train.npy')
y_test = np.load(PATH + PREFIX + 'y_test.npy')
print(x_train.shape, y_train.shape)  # ? (8005, 170, 170, 3) (8005,)
print(x_test.shape, y_test.shape)  # ? (2023, 170, 170, 3) (2023,)

INPUT_SHAPE = x_train.shape[1:]
OUTPUT_DIM = 1  # 이진분류 -> sigmoid (y 는 0/1 숫자 라벨)

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
model.add(Dense(OUTPUT_DIM, activation='sigmoid'))

model.summary()

#! 3. compile & fit -------------------------------------------------------------
# val_loss 가 PATIENCE 에폭 동안 좋아지지 않으면 멈춘다
model.compile(
    loss='binary_crossentropy',
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
y_predict = np.round(model.predict(x_test)).ravel()  # sigmoid 확률 -> 0/1
acc = accuracy_score(y_test, y_predict)  # y_test 는 이미 0/1 숫자 라벨

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
| subject=cat_dog_vgg16 | seed=333 | trainable=False | dense=64 | vs=0.15 | bs=64 | ep=100 | lr=0.001 | pat=10
|| loss=0.2308 | acc=0.9076 | stop=10 | time=209.0s | 20898.5ms/epoch |

"""
