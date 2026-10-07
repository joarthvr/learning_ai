# [학습 정리] keras68 Conv1D diabetes (keras42_CNN_02 의 Conv2D 를 Conv1D 로 바꾼 것)

import time

from sklearn.datasets import load_diabetes
from sklearn.metrics import root_mean_squared_error
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
    {'filters': 16, 'dropout': 0.2},
    {'filters': 32, 'dropout': 0.2},
    {'filters': 64, 'dropout': 0.2},
    {'filters': 32, 'dropout': 0.2},
    {'filters': 16, 'dropout': 0.2},
]
KERNEL = 3
POOL = 2
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.001
PATIENCE = 20  # val_loss 가 이만큼 연속으로 개선되지 않으면 멈춘다

set_random_seed(SEED)
SUBJECT = 'diabetes'
# ======================================================================

#! 1. 데이터 -------------------------------------------------------------
datasets = load_diabetes()
x = datasets.data
y = datasets.target

INPUT_DIM = x.shape[1]  # 10
OUTPUT_DIM = 1  # 회귀

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
)

# ======================================================================
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # fit 은 train 에만 (데이터 누수 방지)

# 스케일링 뒤에 3차원으로 (Conv1D 는 (N, steps, channels) 를 받는다)
x_train = x_train.reshape(-1, INPUT_DIM, 1)
x_test = x_test.reshape(-1, INPUT_DIM, 1)
# ======================================================================

#!  2. 모델 구성 (Conv1D) -------------------------------------------------------------
model = Sequential()
model.add(Input(shape=(INPUT_DIM, 1)))

for layer in LAYERS:
    model.add(Conv1D(layer['filters'], KERNEL, activation=ACTIVATION, padding='same'))
    dropout = layer.get('dropout', 0)
    if dropout > 0:
        model.add(Dropout(dropout))

model.add(MaxPool1D(POOL))  # 10 -> 5
model.add(GlobalAveragePooling1D())
model.add(Dense(OUTPUT_DIM, activation='relu'))

model.summary()

#!  3. 컴파일 훈련 -------------------------------------------------------------
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
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

#! 4. 결과 예측 -------------------------------------------------------------
rmse = root_mean_squared_error(y_test, model.predict(x_test))

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
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={stopped}/{EPOCHS} | pat={PATIENCE} '
    f'| lr={LEARNING_RATE} '
    f'|| rmse={rmse:.4f} '
    f'| time={took:.1f}s | {took / stopped * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
[참고] keras42_CNN_02 (Conv2D) : rmse=59.4073 | time=16.4s

===== RESULT =====

| subject=diabetes | seed=42 | filters=16-32-64-32-16 | act=relu | do=0.2-0.2-0.2-0.2-0.2 | k=3 | pool=2 | ts=0.8 | vs=0.15 | bs=10 | ep=60/100 | pat=20 | lr=0.001
|| rmse=56.9603 | time=12.5s | 209.0ms/epoch |

| subject=diabetes | seed=42 | filters=32-64 | act=relu | do=0.0-0.25 | k=3 | pool=2 | ts=0.8 | vs=0.15 | bs=10 | ep=92/100 | pat=20 | lr=0.001
|| rmse=63.4404 | time=27.4s | 298.2ms/epoch |

| subject=diabetes | seed=42 | filters=32-64-32-16 | act=relu | do=0.2-0.2-0.2-0.2 | k=3 | pool=2 | ts=0.8 | vs=0.15 | bs=10 | ep=60/100 | pat=20 |lr=0.001
|| rmse=60.2673 | time=11.6s | 193.3ms/epoch |
===================================
"""
