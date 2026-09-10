import time

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 77

# --- 데이터 ---
TRAIN_SIZE = 0.7
VAL_SPLIT = 0.15

# --- 모델 ---
HIDDEN_UNITS = [64, 32, 16]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 100
BATCH_SIZE = 42
LEARNING_RATE = 0.001
PATIENCE = 15

set_random_seed(SEED)
# ======================================================================

# 1. 데이터
DATA_PATH = './_data/kaggle_bike/'

train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
submission = pd.read_csv(DATA_PATH + 'sampleSubmission.csv', index_col=0)

# casual + registered = count 이므로 남겨두면 답을 보고 푸는 셈 (타깃 누수)
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

INPUT_DIM = x.shape[1]  # 8
OUTPUT_DIM = 1  # 회귀: 숫자 1개

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
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
model.add(Dense(OUTPUT_DIM))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

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
rmse = root_mean_squared_error(y_test, model.predict(x_test))

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
    f'|| rmse={rmse:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')

"""
===== RESULT =====
| seed=77 | units=64-32-16 | act=relu | ts=0.7 | vs=0.15 | bs=42 | ep=100 | lr=0.001 | pat=15
|| rmse=151.1284| stop=ES미발동 | time=16.7ss |
===================================
"""
