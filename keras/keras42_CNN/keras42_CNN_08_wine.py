import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---
# CNN: 피처 13개를 (13, 1, 1) 세로 벡터로 보고 세로 방향으로만 합성곱한다.
LAYERS = [
    {'filters': 32, 'dropout': 0.0},
    {'filters': 64, 'dropout': 0.25},
    {'filters': 128, 'dropout': 0.25},
]
KERNEL = (3, 1)  # 너비가 1 이라 가로는 1 로 고정
POOL = (2, 1)
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.001

set_random_seed(SEED)
SUBJECT = 'wine'
# ======================================================================

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = pd.get_dummies(datasets.target, dtype='float32')  # 원핫 (178, 3)

INPUT_DIM = x.shape[1]  # 13
OUTPUT_DIM = y.shape[1]  # 3

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y,  # 클래스 비율 유지
)

# ======================================================================
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # fit 은 train 에만 (데이터 누수 방지)

# 스케일링 뒤에 4차원으로 (Conv2D 는 (N, H, W, C) 를 받는다)
x_train = x_train.reshape(-1, INPUT_DIM, 1, 1)
x_test = x_test.reshape(-1, INPUT_DIM, 1, 1)

print(x_train.shape, y_train.shape)  # (142, 13, 1, 1) (142, 3)
print(x_test.shape, y_test.shape)  # (36, 13, 1, 1) (36, 3)
# ======================================================================


# 2. 모델 구성 (CNN)
model = Sequential()
model.add(Input(shape=(INPUT_DIM, 1, 1)))

for layer in LAYERS:
    model.add(Conv2D(layer['filters'], KERNEL, activation=ACTIVATION, padding='same'))
    dropout = layer.get('dropout', 0)
    if dropout > 0:
        model.add(Dropout(dropout))

model.add(MaxPool2D(POOL))  # 13 -> 6
model.add(GlobalAveragePooling2D())
model.add(Dense(OUTPUT_DIM, activation='softmax'))  # 다중분류

model.summary()

# 3. 컴파일 훈련 (EarlyStopping 없이 EPOCHS 를 끝까지 돌린다)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()
hist = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=VAL_SPLIT,
)
end_time = time.time()

# 4. 결과 예측
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 -> 라벨
y_test_label = np.argmax(y_test, axis=-1)  # 원핫 -> 라벨
acc = accuracy_score(y_test_label, y_predict)

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(str(layer['filters']) for layer in LAYERS)
dropouts = '-'.join(str(layer.get('dropout', 0)) for layer in LAYERS)
took = end_time - start_time

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT}   | seed={SEED} '
    f'| filters={structure} | act={ACTIVATION} | do={dropouts} '
    f'| k={KERNEL[0]}x{KERNEL[1]} | pool={POOL[0]}x{POOL[1]} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} '
    f'|| acc={acc:.4f} '
    f'| time={took:.1f}s | {took / EPOCHS * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
===== RESULT ===== cpu
| subject=wine | device=CPU | seed=42 | units=64-32 | act=relu | do=0.2-0.2 | ts=0.8 | vs=0.15 | bs=4 | ep=100| lr=0.001
|| acc=1.0000 | time=12.0s | 120.2ms/epoch |
===================================
===== RESULT ===== gpu
| subject=wine | device=GPU | seed=42 | units=64-32 | act=relu | do=0.2-0.2 | ts=0.8 | vs=0.15 | bs=4 | ep=100 | lr=0.001
|| acc=1.0000 | time=16.0s | 159.7ms/epoch |
===================================

"""
