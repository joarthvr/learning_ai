"""저장해둔 npy 를 불러와 학습한다 (짝: keras45_03_catdog_save_npy.py).

PATH_NPY / PREFIX / IMG_SIZE 는 짝인 save 파일과 정확히 같아야 한다.
다르면 엉뚱한 배열을 읽거나 FileNotFoundError 가 난다.
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
IMG_SIZE = (170, 170)
CHANNELS = 3  # 컬러
PATH_NPY = './_data/kaggle_cat_dog_npy/'
PREFIX = 'keras45_03_'  # save 쪽과 반드시 같아야 한다
VAL_SPLIT = 0.2

# --- 모델 ---
LAYERS = [
    {'filters': 32, 'blocks': 2},
    {'filters': 64, 'blocks': 2},
    {'filters': 128, 'blocks': 2},
]
KERNEL = (3, 3)
ACTIVATION = 'relu'
DROPOUT = 0.2
DENSE_UNITS = [64, 32]

# --- 훈련 ---
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 15

set_random_seed(SEED)
SUBJECT = 'cat_dog'
# ======================================================================

#! 1. 데이터 -----------------------------------------------------------
#! npy 라 디스크에서 바로 읽는다. flow_from_directory 처럼 리사이즈를 다시 하지 않는다.
x_train = np.load(PATH_NPY + PREFIX + 'x_train.npy')
y_train = np.load(PATH_NPY + PREFIX + 'y_train.npy')
x_test = np.load(PATH_NPY + PREFIX + 'x_test.npy')
y_test = np.load(PATH_NPY + PREFIX + 'y_test.npy')

print(x_train.shape, y_train.shape)  # ? (8005, 100, 100, 3) (8005,)
print(x_test.shape, y_test.shape)  # ? (2023, 100, 100, 3) (2023,)

OUTPUT_DIM = 1  # 이진분류

#! 2. 모델 구성 -----------------------------------------------------------
inputs = Input(shape=(*IMG_SIZE, CHANNELS))

x = inputs
for layer in LAYERS:
    for _ in range(layer['blocks']):
        x = Conv2D(layer['filters'], KERNEL, activation=ACTIVATION, padding='same')(x)
        x = Conv2D(layer['filters'], KERNEL, activation=ACTIVATION, padding='same')(x)
    x = MaxPool2D()(x)
    x = Dropout(DROPOUT)(x)

x = GlobalAveragePooling2D()(x)
for units in DENSE_UNITS:
    x = Dense(units, activation=ACTIVATION)(x)
x = Dropout(DROPOUT)(x)
outputs = Dense(OUTPUT_DIM, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

#! 3. 컴파일, 훈련 -----------------------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min', restore_best_weights=True)

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
#! batch_size 를 주지 않으면 x_test 전체를 GPU 로 복사하다 OOM 이 난다.
loss = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
loss_value = loss[0] if isinstance(loss, (list, tuple)) else loss
y_predict = np.round(model.predict(x_test, batch_size=BATCH_SIZE))  # 0.5 기준 -> 0 또는 1
acc = accuracy_score(y_test, y_predict)  # y_test 는 원핫이 아니라 0/1 라벨

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(f'{layer["filters"]}x{layer["blocks"]}' for layer in LAYERS)
dense = '-'.join(str(u) for u in DENSE_UNITS)
took = end_time - start_time

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT} | seed={SEED} '
    f'| img={IMG_SIZE[0]}x{IMG_SIZE[1]} | conv={structure} | k={KERNEL[0]}x{KERNEL[1]} '
    f'| dense={dense} | do={DROPOUT} '
    f'| vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} | pat={PATIENCE} '
    f'|| acc={acc:.4f} | loss={loss_value:.4f} '
    f'| time={took:.1f}s |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
===== RESULT =====
| subject=cat_dog | seed=42 | img=150x150 | conv=32x2-64x2-128x2 | k=3x3 | dense=64-32 | do=0.2| vs=0.2 | bs=32 | ep=200 | pat=15
|| acc=0.8413 | loss=0.4126 | time=622.3s |
===================================
"""
