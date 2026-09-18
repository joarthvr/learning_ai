import time

import pandas as pd
from sklearn.metrics import root_mean_squared_error
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
HIDDEN_UNITS = [64, 32, 16, 8]
ACTIVATION = 'relu'
DROPOUT = 0.3
# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
SUBJECT = 'ddarung'
# ======================================================================

# 1. 데이터
DATA_PATH = './_data/ddarung/'

train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
submission = pd.read_csv(DATA_PATH + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()  # 결측치 제거

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

INPUT_DIM = x.shape[1]  # 9
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
    # Dropout 은 Dense 바로 뒤에 붙인다. 훈련 중에만 뉴런을 무작위로 끄고
    # 평가/예측 때는 자동으로 꺼져서 전체 뉴런을 쓴다 (직접 끌 필요 없음)
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
======================= 실험 기록 =======================
===== RESULT ===== 0.2
| subject=ddarung | seed=42 | units=64-32-16-8 | act=relu | do=0.2 | ts=0.8 | vs=0.15 | bs=8 | ep=1000 | lr=0.001 | pat=20
|| rmse=44.6605 | stop=194 | time=49.7s |
===================================
===== RESULT ===== 0.3
| subject=ddarung | seed=42 | units=64-32-16-8 | act=relu | do=0.3 | ts=0.8 | vs=0.15 | bs=8 | ep=1000 | lr=0.001 | pat=20
|| rmse=46.3558 | stop=211 | time=57.0s |
===================================
"""
