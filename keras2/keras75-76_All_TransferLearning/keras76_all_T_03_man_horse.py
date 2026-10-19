import warnings

warnings.filterwarnings('ignore')

import random
import time

import numpy as np
import tensorflow as tf
from keras.applications import (
    VGG19,
    DenseNet201,
    MobileNetV2,
    ResNet50,
    ResNet101,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# ======================================================================
SEED = 333
VAL_SPLIT = 0.15

# --- 모델 ---
TRAINABLE = False
DENSE_UNITS = 64

# --- 훈련 ---
EPOCHS = 1000
PATIENCE = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

SUBJECT = 'man_horse'
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#! 1. 데이터 ------------------------------------------------------
x_train = np.load('./_save/keras46/horse/keras46_horse_x_train.npy')
x_test = np.load('./_save/keras46/horse/keras46_horse_x_test.npy')
y_train = np.load('./_save/keras46/horse/keras46_horse_y_train.npy')
y_test = np.load('./_save/keras46/horse/keras46_horse_y_test.npy')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

INPUT_SHAPE = x_train.shape[1:]
OUTPUT_DIM = y_train.shape[1]  # 원핫이라 클래스 수 = 2

# 이미지 크기가 데이터마다 달라서 데이터를 불러온 뒤에 모델을 선언한다
model_list = [
    VGG19(include_top=False, input_shape=INPUT_SHAPE),
    ResNet50(include_top=False, input_shape=INPUT_SHAPE),
    ResNet101(include_top=False, input_shape=INPUT_SHAPE),
    DenseNet201(include_top=False, input_shape=INPUT_SHAPE),
    MobileNetV2(include_top=False, input_shape=INPUT_SHAPE),
]

for model_i in model_list:
    model_i.trainable = TRAINABLE

    #! 2. model
    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(OUTPUT_DIM, activation='softmax'))

    #! 3. compile & fit
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc'],
    )

    es = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
    )

    rlr = ReduceLROnPlateau(
        monitor='loss',
        verbose=0,
        patience=PATIENCE,
    )

    start_time = time.time()
    hist = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=VAL_SPLIT,
        callbacks=[es, rlr],
    )
    end_time = time.time()

    #! 4. evaluate, predict
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('----------------------------------------------------')
    print('모델명: ', model_i.name, 'loss: ', loss[0], 'acc: ', loss[1])
    print('----------------------------------------------------')
