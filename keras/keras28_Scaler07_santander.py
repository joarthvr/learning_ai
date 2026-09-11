import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---
HIDDEN_UNITS = [128, 64, 32, 16]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 2048
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
# ======================================================================

# 1. 데이터
DATA_PATH = './_data/kaggle_santander/'

train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
submission = pd.read_csv(DATA_PATH + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

INPUT_DIM = x.shape[1]  # 200
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
# scaler = MaxAbsScaler()
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

# target 이 90:10 이라 acc 는 전부 0으로 찍어도 0.90 -> AUC 를 같이 볼 것
auc = roc_auc_score(y_test, y_prob)

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
    f'|| acc={acc:.4f} | auc={auc:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')

"""

===== RESULT ===== minmax
| seed=42 | units=128-64-32-16 | act=relu | ts=0.8 | vs=0.15 | bs=2048 | ep=1000 | lr=0.001 | pat=20
|| acc=0.9136 | auc=0.8599 | stop=53 | time=14.7s |
===================================

===== RESULT ===== abs
| seed=42 | units=128-64-32-16 | act=relu | ts=0.8 | vs=0.15 | bs=2048 | ep=1000 | lr=0.001 | pat=20
|| acc=0.9105 | auc=0.8481 | stop=30 | time=8.8s |
===================================

===== RESULT ===== standard
| seed=42 | units=128-64-32-16 | act=relu | ts=0.8 | vs=0.15 | bs=2048 | ep=1000 | lr=0.001 | pat=20
|| acc=0.9082 | auc=0.8417 | stop=22 | time=6.5s |
===================================
"""
