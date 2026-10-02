import time

from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---
# LSTM: 입력을 시간 순서(timesteps)로 한 칸씩 읽는다. (samples, timesteps, features)

LAYERS = [
    {'units': 64, 'dropout': 0.2},
    {'units': 32, 'dropout': 0.2},
    {'units': 16, 'dropout': 0.2},
    {'units': 8, 'dropout': 0.2},
]

# --- 훈련 ---
EPOCHS = 100
PATIENCE = 10  # EarlyStopping
BATCH_SIZE = 4
LEARNING_RATE = 0.001

set_random_seed(SEED)
SUBJECT = 'boston'
# ======================================================================

# 1. 데이터
# boston_housing 은 이미 train/test 로 나뉘어 나온다
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(
    test_split=1 - TRAIN_SIZE,
    seed=SEED,
)

INPUT_DIM = x_train.shape[1]  # 13
OUTPUT_DIM = 1  # 회귀

SEQ_SHAPE = (INPUT_DIM, 1)  # 피처 하나하나가 한 시점, 시점당 값은 1개

# ======================================================================
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # fit 은 train 에만 (데이터 누수 방지)

# 스케일링 뒤에 3차원으로 (LSTM 은 (N, timesteps, features) 를 받는다)
x_train = x_train.reshape(-1, *SEQ_SHAPE)
x_test = x_test.reshape(-1, *SEQ_SHAPE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# ======================================================================

# 2. 모델 구성 (LSTM)
model = Sequential()
model.add(Input(shape=SEQ_SHAPE))

for i, layer in enumerate(LAYERS):
    # 뒤에 LSTM 이 또 오면 시퀀스를 그대로 넘겨야 하므로 마지막 층만 False
    is_last = i == len(LAYERS) - 1
    model.add(LSTM(layer['units'], return_sequences=not is_last))
    dropout = layer.get('dropout', 0)
    if dropout > 0:
        model.add(Dropout(dropout))

model.add(Dense(OUTPUT_DIM))  # 회귀: 활성화 없음

model.summary()

# 3. 컴파일 훈련 (val_loss 가 PATIENCE 에폭 동안 좋아지지 않으면 멈춘다)
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,  # 가장 좋았던 에폭의 가중치로 되돌린다
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
structure = '-'.join(str(layer['units']) for layer in LAYERS)
dropouts = '-'.join(str(layer.get('dropout', 0)) for layer in LAYERS)
took = end_time - start_time

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT} | seed={SEED} '
    f'| lstm={structure} | do={dropouts} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| rmse={rmse:.4f} '
    f'| stop={es.stopped_epoch if es.stopped_epoch else "ES미발동"} '
    f'| time={took:.1f}s | {took / (es.stopped_epoch or EPOCHS) * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
"""
