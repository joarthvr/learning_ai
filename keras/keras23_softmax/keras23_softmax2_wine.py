import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

# 목표: acc > 0.95
# ======================================================================
SEED = 156  # 난수 고정 (재현성) — 튜닝 대상 아님!

# --- 데이터 ---
TRAIN_SIZE = 0.8  # train / test 분할 비율
VAL_SPLIT = 0.15  # train 중 검증에 쓸 비율

# --- 모델 ---
HIDDEN_UNITS = [64, 32]  # 은닉층 구조 (리스트 길이 = 층 수)
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 4
LEARNING_RATE = 0.001  # Adam 기본값
PATIENCE = 20
# ======================================================================

set_random_seed(SEED)  # python / numpy / tensorflow 난수를 한 번에 고정


# 1. 데이터
datasets = load_wine()

x = datasets.data
y = pd.get_dummies(datasets.target, dtype='float32')

print(x.shape, y.shape)  # (178, 13) (178, 3)
print(np.unique(datasets.target, return_counts=True))  # (array([0,1,2]), array([59,71,48]))

# 데이터에서 유도되는 값 — 하이퍼파라미터가 아니므로 위 블록에 두지 않는다
INPUT_DIM = x.shape[1]  # 13
N_CLASSES = y.shape[1]  # 3

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
stop_ep = es.stopped_epoch if es.stopped_epoch else 'ES미발동'
took = end_time - start_time

print('')
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
===== RESULT =====
| seed=156 | units=64-32 | act=relu | ts=0.8 | vs=0.15 | bs=4 | ep=1000 | lr=0.001 | pat=20
|| train=0.8380 | test=0.8333 | stop=35 | time=3.1s |
===================================
===== RESULT =====
| seed=156 | units=64-32 | act=relu | ts=0.8 | vs=0.15 | bs=4 | ep=1000 | lr=0.001 | pat=20
|| train=0.8380 | test=0.8333 | stop=35 | time=2.9s |
===================================

"""
