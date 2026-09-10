import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42  # 난수 고정 (재현성) — 튜닝 대상 아님!

# --- 데이터 ---
TRAIN_SIZE = 0.8  # train / test 분할 비율
VAL_SPLIT = 0.15  # train 중 검증에 쓸 비율

# --- 모델 ---
HIDDEN_UNITS = [128, 64, 32, 16]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 2048
LEARNING_RATE = 0.001  # Adam 기본값
PATIENCE = 20
# ======================================================================

set_random_seed(SEED)  # python / numpy / tensorflow 난수를 한 번에 고정

DATA_PATH = './_data/kaggle_santander/'

# 1. 데이터
train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
submission = pd.read_csv(DATA_PATH + 'sample_submission.csv', index_col=0)
x = train_csv.drop(['target'], axis=1)
# 이진 문제지만 다중분류(softmax) 방식으로 푼다 -> 정답도 원핫이어야 함
y = pd.get_dummies(train_csv['target'], dtype='float32')  # (200000, 2)


INPUT_DIM = x.shape[1]
N_CLASSES = y.shape[1]  # 2 (원핫이라 열 개수 = 클래스 수)
print(train_csv.shape)  # (200000, 201)
print(test_csv.shape)  # (200000, 200)
print(submission.shape)  # (200000, 1)

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
y_prob = model.predict(x_test)  # (60000, 2) 각 클래스 확률
y_predict = np.argmax(y_prob, axis=-1)  # 확률 -> 라벨. axis 없으면 스칼라 1개가 됨
y_test_label = np.argmax(y_test, axis=-1)  # 원핫 정답 -> 라벨
acc = accuracy_score(y_test_label, y_predict)

# target 비율이 90:10 이라 acc는 전부 0으로 찍어도 0.90 이 나옴 -> AUC를 같이 볼 것
auc = roc_auc_score(y_test_label, y_prob[:, 1])  # 양성(1) 클래스 확률만

train_acc = model.evaluate(x_train, y_train, verbose=0)[1]
test_acc = model.evaluate(x_test, y_test, verbose=0)[1]

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(map(str, HIDDEN_UNITS))
stop_ep = es.stopped_epoch if es.stopped_epoch else 'ES미발동'
took = end_time - start_time

#####################################################
y_submit = model.predict(test_csv)
submission['target'] = y_submit.round()
submission.to_csv(DATA_PATH + 'submit/' + 'submit_0910_1.csv')

print('')
print('===== RESULT =====')
print(
    f'| seed={SEED} | units={structure} | act={ACTIVATION} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| train={train_acc:.4f} | test={test_acc:.4f} | auc={auc:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')


"""
===== RESULT =====
| seed=42 | units=128-64-32-16 | act=relu | ts=0.8 | vs=0.15 | bs=2048 | ep=1000 | lr=0.001 |
pat=20 || train=0.9123 | test=0.9105 | auc=0.8377 | stop=28 | time=49.6s |
===================================
===== RESULT =====
| seed=42 | units=128-64-32-16 | act=relu | ts=0.8 | vs=0.15 | bs=2048 | ep=1000 | lr=0.001 |
pat=20 || train=0.9127 | test=0.9103 | auc=0.8384 | stop=29 | time=8.6s |
===================================
===== RESULT =====
| seed=42 | units=64-32-16 | act=relu | ts=0.7 | vs=0.25 | bs=1024
| ep=1000| lr=0.001 | pat=20 || train=0.9102 | test=0.9081 | auc=0.8367 | stop=30 | time=7.9s |
===================================
===== RESULT =====
| seed=42 | units=128-64-32-16 | act=relu | ts=0.7 | vs=0.25 | bs=1024 | ep=1000 | lr=0.001 |
pat=20 || train=0.1005 | test=0.1005 | auc=0.5000 | stop=20 | time=6.3s |
===================================
===== RESULT =====
| seed=42 | units=128-64-32-16 | act=relu | ts=0.8 | vs=0.15 | bs=2048 | ep=1000 | lr=0.001 |
pat=20 || train=0.1005 | test=0.1005 | auc=0.5000 | stop=20 | time=6.0s |
===================================

"""
