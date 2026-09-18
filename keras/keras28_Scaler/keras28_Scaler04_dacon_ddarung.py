import time

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 153

# --- 데이터 ---
TRAIN_SIZE = 0.7
VAL_SPLIT = 0.3

# --- 모델 ---
HIDDEN_UNITS = [12, 12, 8]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 200
BATCH_SIZE = 42
LEARNING_RATE = 0.001
PATIENCE = 15

set_random_seed(SEED)
# ======================================================================

# 1. 데이터
DATA_PATH = './_data/ddarung/'

train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
submission = pd.read_csv(DATA_PATH + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()  # 결측치 제거 -> [1328 rows x 10 columns]

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

INPUT_DIM = x.shape[1]  # 9
OUTPUT_DIM = 1  # 회귀: 숫자 1개

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
)

# ======================================================================
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # train 에서 구한 Min/Max 를 그대로 적용
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
===== RESULT ===== minmax
| seed=153 | units=12-12-8 | act=relu | ts=0.7 | vs=0.3 | bs=42 | ep=200 | lr=0.001 | pat=15
|| rmse=51.2163 | stop=ES미발동 | time=12.0s |
===================================

===== RESULT ===== standard
| seed=153 | units=12-12-8 | act=relu | ts=0.7 | vs=0.3 | bs=42 | ep=200 | lr=0.001 |pat=15
|| rmse=49.6318 | stop=ES미발동 | time=11.9s |
===================================

===== RESULT =====
| seed=153 | units=12-12-8 | act=relu | ts=0.7 | vs=0.3 | bs=42 | ep=200 | lr=0.001 |pat=15
|| rmse=51.3603 | stop=ES미발동 | time=12.4s |
===================================

===== RESULT ===== robust
| seed=153 | units=12-12-8 | act=relu | ts=0.7 | vs=0.3 | bs=42 | ep=200 | lr=0.001 | pat=15
|| rmse=50.2420 | stop=ES미발동 | time=11.9s |
===================================
"""
