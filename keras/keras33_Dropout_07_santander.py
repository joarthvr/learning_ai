import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---
HIDDEN_UNITS = [64, 32, 16]
ACTIVATION = 'relu'
DROPOUT = 0.3  # 은닉층마다 끌 뉴런 비율. 0 이면 드롭아웃 없음 (0.2~0.5 가 보통)

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
SUBJECT = 'santander'
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
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # fit 은 train 에만 (데이터 누수 방지)
# ======================================================================

# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(INPUT_DIM,)))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation=ACTIVATION))
    # Dropout 은 Dense 바로 뒤에 붙인다. 훈련 중에만 뉴런을 무작위로 끄고
    # 평가/예측 때는 자동으로 꺼져서 전체 뉴런을 쓴다 (직접 끌 필요 없음)
    if DROPOUT > 0:
        model.add(Dropout(DROPOUT))
model.add(Dense(OUTPUT_DIM, activation='sigmoid'))  # 출력층 뒤에는 Dropout 을 붙이지 않는다

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
    f'| subject={SUBJECT} | seed={SEED} | units={structure} | act={ACTIVATION} '
    f'| do={DROPOUT} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| acc={acc:.4f} | auc={auc:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
===== RESULT =====
| subject=santander | seed=42 | units=64-32-16 | act=relu | do=0.2 | ts=0.8 | vs=0.15 | bs=1024 | ep=1000 | lr=0.001 | pat=20
|| acc=0.9105 | auc=0.8509 | stop=29 |time=19.8s |
===================================
===== RESULT =====
| subject=santander | seed=42 | units=64-32-16 | act=relu | do=0.3 | ts=0.8 | vs=0.15 | bs=1024 | ep=1000 | lr=0.001 | pat=20
|| acc=0.9114 | auc=0.8522 | stop=32 |time=22.3s |
===================================

"""
