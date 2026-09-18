import time

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
HIDDEN_UNITS = [64, 32, 16, 8]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
# ======================================================================

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

INPUT_DIM = x.shape[1]  # 8
OUTPUT_DIM = 1

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
)

# ======================================================================
# MinMaxScaler: (원값 - Min) / (Max - Min)  -> 모든 열을 0~1 로
scaler = MinMaxScaler()
# scaler = StandardScaler()
# fit 은 train 에만! test 에 fit 하면 test 정보가 새어나감(데이터 누수)
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
===== RESULT ===== // 민맥스
| seed=42 | units=32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=32 | ep=1000
| lr=0.001 | pat=20 || rmse=0.5555 | stop=199 | time=74.1s |
===================================

===== RESULT ===== // 민맥스
| seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=16 | ep=1000
| lr=0.001 | pat=20 || rmse=0.5673 | stop=71 | time=54.3s |
===================================

===== RESULT ===== // 스탠다드
| seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=16 | ep=1000 | lr=0.001 | pat=20
|| rmse=0.5355 | stop=90 | time=63.8s |
===================================

===== RESULT ===== // abs
| seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=16 | ep=1000 | lr=0.001 | pat=20
|| rmse=0.6088 | stop=152 | time=109.4s |
===================================

"""
