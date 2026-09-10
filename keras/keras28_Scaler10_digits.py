import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.7
VAL_SPLIT = 0.25

# --- 모델 ---
HIDDEN_UNITS = [64, 32, 16]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 72
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
# ======================================================================

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = pd.get_dummies(datasets.target, dtype='float32')  # 원핫 (1797, 10)

INPUT_DIM = x.shape[1]  # 64 (8x8 이미지를 펼친 것)
OUTPUT_DIM = y.shape[1]  # 10

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y,  # 클래스 비율 유지
)

# ======================================================================
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# ======================================================================

# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(INPUT_DIM,)))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation=ACTIVATION))
model.add(Dense(OUTPUT_DIM, activation='softmax'))

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
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

# 4. 결과 예측
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 -> 라벨
y_test_label = np.argmax(y_test, axis=-1)  # 원핫 -> 라벨
acc = accuracy_score(y_test_label, y_predict)

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
    f'|| acc={acc:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')

"""
===== RESULT =====
| seed=42 | units=64-32-16 | act=relu | ts=0.7 | vs=0.25 | bs=72 | ep=1000 | lr=0.001 | pat=20
|| acc=0.9574 | stop=104 | time=6.8s |
===================================
"""
