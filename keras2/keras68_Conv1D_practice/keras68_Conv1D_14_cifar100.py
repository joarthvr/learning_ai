# [학습 정리] keras68 Conv1D cifar100 (keras52_optimizer_14 의 Conv2D 를 Conv1D 로 바꾼 것)

import time

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed, to_categorical

# ======================================================================
SEED = 42

# --- 데이터 ---
VAL_SPLIT = 0.2

# --- 모델 ---
IMG_SIZE = (32, 32 * 3)  # (steps, channels) = (행, 가로 x RGB)
LAYERS = [
    {'filters': 64, 'dropout': 0.0},
    {'filters': 128, 'dropout': 0.25},
    {'filters': 128, 'dropout': 0.25},
]
KERNEL = 3
POOL = 2
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 5  # val_loss 가 이만큼 연속으로 개선되지 않으면 멈춘다

set_random_seed(SEED)
SUBJECT = 'cifar100'
# ======================================================================

#! 1. 데이터 -------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 0~1 로 맞추고, 가로와 RGB 를 한 축으로 합친다: (N, 32, 32, 3) -> (N, 32, 96)
x_train = (x_train / 255.0).reshape(-1, *IMG_SIZE)
x_test = (x_test / 255.0).reshape(-1, *IMG_SIZE)

OUTPUT_DIM = 100
y_train = to_categorical(y_train, OUTPUT_DIM)  # 원핫 (50000, 100)
y_test = to_categorical(y_test, OUTPUT_DIM)

print(x_train.shape, y_train.shape)  # (50000, 32, 96) (50000, 100)
print(x_test.shape, y_test.shape)  # (10000, 32, 96) (10000, 100)

#! 2. 모델 구성 (Conv1D) -------------------------------------------------------------
model = Sequential()
model.add(Input(shape=IMG_SIZE))

for layer in LAYERS:
    model.add(Conv1D(layer['filters'], KERNEL, activation=ACTIVATION, padding='same'))
    dropout = layer.get('dropout', 0)
    if dropout > 0:
        model.add(Dropout(dropout))

model.add(MaxPool1D(POOL))  # 32 -> 16
model.add(GlobalAveragePooling1D())
model.add(Dense(OUTPUT_DIM, activation='softmax'))  # 다중분류

model.summary()

#! 3. 컴파일 훈련 -------------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(patience=PATIENCE, restore_best_weights=True)

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

#! 4. 결과 예측 -------------------------------------------------------------
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 -> 라벨
y_test_label = np.argmax(y_test, axis=-1)  # 원핫 -> 라벨
acc = accuracy_score(y_test_label, y_predict)

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(str(layer['filters']) for layer in LAYERS)
dropouts = '-'.join(str(layer.get('dropout', 0)) for layer in LAYERS)
took = end_time - start_time
stopped = len(hist.history['loss'])  # ES 로 멈춘 실제 에폭 수

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT} | seed={SEED} '
    f'| filters={structure} | act={ACTIVATION} | do={dropouts} '
    f'| k={KERNEL} | pool={POOL} '
    f'| vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={stopped}/{EPOCHS} | pat={PATIENCE} '
    f'| lr={LEARNING_RATE} '
    f'|| acc={acc:.4f} '
    f'| time={took:.1f}s | {took / stopped * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
[참고] keras52_optimizer_14 (Conv2D, 증폭 30000장 + ES) : acc=0.4564 | time=257.3s
"""
