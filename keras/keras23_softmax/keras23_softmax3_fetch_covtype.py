import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

# 목표: acc > 0.93
# ======================================================================
SEED = 153  # 난수 고정 (재현성) — 튜닝 대상 아님!

# --- 데이터 ---
TRAIN_SIZE = 0.7  # train / test 분할 비율
VAL_SPLIT = 0.25  # train 중 검증에 쓸 비율

# --- 모델 ---
HIDDEN_UNITS = [1024, 512, 256, 128, 64, 32]  # 은닉층 구조 (개수 = 층 수)
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 2048
LEARNING_RATE = 0.001  # Adam 기본값
PATIENCE = 100
# ======================================================================

set_random_seed(SEED)  # python / numpy / tensorflow 난수를 한 번에 고정


# 1. 데이터
datasets = fetch_covtype()

x = datasets.data
y = pd.get_dummies(datasets.target, dtype='float32')

print(x.shape, y.shape)  # (581012, 54) (581012, 7)
print(np.unique(datasets.target, return_counts=True))  # 라벨 1~7, 불균형 확인

# 데이터에서 유도되는 값 — 하이퍼파라미터가 아니므로 위 블록에 두지 않는다
INPUT_DIM = x.shape[1]  # 54
N_CLASSES = y.shape[1]  # 7

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y,
)

# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(INPUT_DIM,)))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation=ACTIVATION))
model.add(Dense(N_CLASSES, activation='softmax'))

# 3. 컴파일 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics=['acc'],
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=PATIENCE,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=[es],
)
end_time = time.time()


# 4. 평가 예측
y_predict = np.argmax(model.predict(x_test), axis=-1)
y_test_label = np.argmax(y_test, axis=-1)
acc = accuracy_score(y_test_label, y_predict)

train_acc = model.evaluate(x_train, y_train, verbose=0)[1]
test_acc = model.evaluate(x_test, y_test, verbose=0)[1]

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(map(str, HIDDEN_UNITS))
print('')
stop_ep = es.stopped_epoch if es.stopped_epoch else 'ES미발동'
took = end_time - start_time

print('===== RESULT =====')
print(
    f'| seed={SEED} | units={structure} | act={ACTIVATION} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| train={train_acc:.4f} | test={test_acc:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
RESULT 한 줄을 그대로 복사해서 아래에 붙여넣기

1) | seed=153 | units=1024-512-256-128-64-32 | act=relu | ts=0.7 | vs=0.25 | bs=2048 | ep=1000 | lr=0.001 | pat=100 || train=0.9652 | test=0.9497 | stop=376 | time=1037.3s |
   -> 목표(0.93) 달성. 다만 17분 소요.

2) | seed=153 | units=1024-512-256-128-64-32 | act=relu | ts=0.7 | vs=0.25 | bs=2048 | ep=1000 | lr=0.001 | pat=?  || train=0.9381 | test=0.9275 | stop=175 | time=297.9s |
   -> patience가 작아 일찍 멈춤. 덜 학습되어 성능 낮음.

3)

"""
