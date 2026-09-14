import time

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---
HIDDEN_UNITS = [64, 32, 16, 8]
DROPOUT_RATIO = [0.3, 0.3, 0.3, 0.3]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 20

PATH_SAVE = './_save/keras30/'

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

# MinMaxScaler: (원값 - Min) / (Max - Min)  -> 모든 열을 0~1 로
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(INPUT_DIM,)))
for idx in range(len(HIDDEN_UNITS)):
    model.add(Dense(HIDDEN_UNITS[idx], activation=ACTIVATION))
    model.add(Dropout(DROPOUT_RATIO[idx]))
model.add(Dense(OUTPUT_DIM))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

# 최적의 weight를 저장
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=PATH_SAVE + 'keras30_mcp1.keras',
    verbose=1,
)

start_time = time.time()
hist = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=VAL_SPLIT,
    callbacks=[es, mcp],
)
end_time = time.time()


# 4.평가, 예측
rmse = root_mean_squared_error(y_test, model.predict(x_test))

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(map(str, HIDDEN_UNITS))
stop_ep = es.stopped_epoch if es.stopped_epoch else 'ES미발동'

print('')
print('===== RESULT =====')
print(
    f'| seed={SEED} | units={structure} | act={ACTIVATION} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| rmse={rmse:.4f} '
    f' time={end_time - start_time}'
)
print('===================================')
print('')

"""

===== RESULT =====
| seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=16 | ep=1000 | lr=0.001 | pat=20
|| rmse=0.5251  time=252.34767246246338
===================================
===== RESULT =====
| seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=16 | ep=1000 | lr=0.001 | pat=20
|| rmse=0.5483  time=203.09300875663757
===================================
===== RESULT =====
| seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=16 | ep=1000 | lr=0.001 | pat=20
|| rmse=0.5552  time=211.96195340156555
===================================
"""
