# [학습 정리] keras68 Conv1D cancer (keras42_CNN_06 의 Conv2D 를 Conv1D 로 바꾼 것)

import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---
LAYERS = [
    {'filters': 32, 'dropout': 0.0},
    {'filters': 64, 'dropout': 0.0},
    {'filters': 64, 'dropout': 0.25},
    {'filters': 64, 'dropout': 0.25},
]
KERNEL = 3
POOL = 2
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 20  # val_loss 가 이만큼 연속으로 개선되지 않으면 멈춘다

set_random_seed(SEED)
SUBJECT = 'cancer'
# ======================================================================

#! 1. 데이터 -------------------------------------------------------------
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

INPUT_DIM = x.shape[1]  # 30
OUTPUT_DIM = 1  # 이진분류

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

# 스케일링 뒤에 3차원으로 (Conv1D 는 (N, steps, channels) 를 받는다)
x_train = x_train.reshape(-1, INPUT_DIM, 1)
x_test = x_test.reshape(-1, INPUT_DIM, 1)
# ======================================================================

#! 2. 모델 구성 (Conv1D) -------------------------------------------------------------
model = Sequential()
model.add(Input(shape=(INPUT_DIM, 1)))

for i, layer in enumerate(LAYERS):
    model.add(Conv1D(layer['filters'], KERNEL, activation=ACTIVATION, padding='same'))
    if i == 1:
        model.add(MaxPool1D(POOL))
    dropout = layer.get('dropout', 0)
    if dropout > 0:
        model.add(Dropout(dropout))

model.add(GlobalAveragePooling1D())
model.add(Dense(OUTPUT_DIM, activation='sigmoid'))

model.summary()

#! 3. 컴파일 훈련 -------------------------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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

# 4. 결과 예측
y_prob = model.predict(x_test)
y_predict = np.round(y_prob)
acc = accuracy_score(y_test, y_predict)

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(str(layer['filters']) for layer in LAYERS)
dropouts = '-'.join(str(layer.get('dropout', 0)) for layer in LAYERS)
took = end_time - start_time
stopped = len(hist.history['loss'])

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT} | seed={SEED} '
    f'| filters={structure} | act={ACTIVATION} | do={dropouts} '
    f'| k={KERNEL} | pool={POOL} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={stopped}/{EPOCHS} | pat={PATIENCE} '
    f'| lr={LEARNING_RATE} '
    f'|| acc={acc:.4f} '
    f'| time={took:.1f}s | {took / stopped * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
[참고] keras42_CNN_06 (Conv2D) : acc=0.9035 | time=20.9s

===== RESULT =====
| subject=cancer | seed=42 | filters=32-64-64-64 | act=relu | do=0.0-0.0-0.25-0.25 | k=3 | pool=2 | ts=0.8 | vs=0.15 | bs=16 | ep=92/100 | pat=20 |lr=0.001
|| acc=0.9386 | time=17.4s | 189.3ms/epoch |
===================================
"""
