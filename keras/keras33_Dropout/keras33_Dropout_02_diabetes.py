import time

from sklearn.datasets import load_diabetes
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 153

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.2

# --- 모델 ---
HIDDEN_UNITS = [64, 32, 16]
ACTIVATION = 'relu'
DROPOUT = 0.2

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 10
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
SUBJECT = 'diabetes'
# ======================================================================

# 1. 데이터
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
# ======================================================================

# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(INPUT_DIM,)))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation=ACTIVATION))
    if DROPOUT > 0:
        model.add(Dropout(DROPOUT))
model.add(Dense(OUTPUT_DIM))  # 출력층 뒤에는 Dropout 을 붙이지 않는다

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
    f'| subject={SUBJECT} | seed={SEED} | units={structure} | act={ACTIVATION} '
    f'| do={DROPOUT} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| rmse={rmse:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')


"""
===== RESULT =====
| subject=diabetes | seed=153 | units=64-32-16 | act=relu | do=0.2 | ts=0.8 | vs=0.2 | bs=10 | ep=1000 | lr=0.001 | pat=20
|| rmse=51.2152 | stop=95 | time=11.4s |
===================================
"""
