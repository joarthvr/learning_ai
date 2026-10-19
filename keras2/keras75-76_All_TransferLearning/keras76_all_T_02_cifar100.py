import warnings

warnings.filterwarnings('ignore')

from keras.applications import (
    VGG19,
    DenseNet201,
    MobileNetV2,
    ResNet50,
    ResNet101,
)

model_list = [
    VGG19(include_top=False, input_shape=(32, 32, 3)),
    ResNet50(include_top=False, input_shape=(32, 32, 3)),
    ResNet101(include_top=False, input_shape=(32, 32, 3)),
    DenseNet201(include_top=False, input_shape=(32, 32, 3)),
    # InceptionV3(include_top=False, input_shape=(32,32,3)),
    # InceptionResNetV2(include_top=False, input_shape=(32,32,3)),
    MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
    # EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
]

import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar100
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
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SUBJECT = 'cifar100'
# ======================================================================


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

for model_i in model_list:
    model_i.trainable = TRAINABLE

    #! 2. model
    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(100, activation='softmax'))

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # 스케일링
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    #! 3. compile & fit
    model.compile(
        loss='sparse_categorical_crossentropy',
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
