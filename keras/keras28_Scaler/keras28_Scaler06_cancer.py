import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 77

# --- 데이터 ---
TRAIN_SIZE = 0.7
VAL_SPLIT = 0.3

# --- 모델 ---
HIDDEN_UNITS = [30, 36, 36, 36]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 150

set_random_seed(SEED)
# ======================================================================

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

INPUT_DIM = x.shape[1]  # 30
OUTPUT_DIM = 1  # 이진분류: 출력 1개 + sigmoid

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y,  # 클래스 비율 유지
)

# ======================================================================
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# ======================================================================

# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(INPUT_DIM,)))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation=ACTIVATION))
model.add(Dense(OUTPUT_DIM, activation='sigmoid'))

# 3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
y_prob = model.predict(x_test)
y_predict = np.round(y_prob)  # 0.5 기준 -> 0 또는 1
acc = accuracy_score(y_test, y_predict)

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
===== RESULT ===== minmax
| seed=77 | units=30-36-36-36 | act=relu | ts=0.7 | vs=0.3 | bs=32 | ep=1000 | lr=0.001 | pat=20
|| acc=0.9649 | stop=41 | time=3.1s |
===================================
===== RESULT ===== minmax
| seed=77 | units=30-36-36-36 | act=relu | ts=0.7 | vs=0.3 | bs=32 | ep=1000 | lr=0.001 | pat=20
|| acc=0.9532 | stop=37 | time=2.9s |
===================================
===== RESULT ===== standard
| seed=77 | units=30-36-36-36 | act=relu | ts=0.7 | vs=0.3 | bs=32 | ep=1000 | lr=0.001 | pat=80
|| acc=0.9532 | stop=97 | time=5.9s |
===================================
===== RESULT ===== standard
| seed=77 | units=30-36-36-36 | act=relu | ts=0.7 | vs=0.3 | bs=32 | ep=1000 | lr=0.001 | pat=150
|| acc=0.9532 | stop=167 | time=9.7s |
===================================
===== RESULT ===== abs
| seed=77 | units=30-36-36-36 | act=relu | ts=0.7 | vs=0.3 | bs=32 | ep=1000 | lr=0.001 | pat=150
|| acc=0.9532 | stop=167 | time=9.6s |
===================================

"""
