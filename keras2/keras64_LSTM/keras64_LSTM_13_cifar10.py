import time

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
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
SEQ_SHAPE = (32, 96)  # 이미지 (32, 32, 3) 을 "32줄 x 줄당 96값(32픽셀 x RGB 3)" 시퀀스로 본다
LAYERS = [
    {'units': 128, 'dropout': 0.2},
    {'units': 64, 'dropout': 0.2},
]

# --- 훈련 ---
EPOCHS = 30
PATIENCE = 5  # EarlyStopping
BATCH_SIZE = 256
LEARNING_RATE = 0.001

set_random_seed(SEED)
SUBJECT = 'cifar10'
# ======================================================================

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

OUTPUT_DIM = 10  # 클래스 수

# 픽셀은 0~255 고정 상수라 train 기준으로 fit 할 필요가 없다 -> 데이터 누수 아님
x_train = x_train / 255.0
x_test = x_test / 255.0

# (N, 이미지 높이, 이미지 너비[, 채널]) -> (N, timesteps, features)
x_train = x_train.reshape(-1, *SEQ_SHAPE)
x_test = x_test.reshape(-1, *SEQ_SHAPE)
print(x_train.shape, x_test.shape)

# y 는 원핫하지 않고 숫자 라벨 그대로 쓴다 -> sparse_categorical_crossentropy
y_train = y_train.ravel()  # (N, 1) -> (N,)  (cifar 는 (N, 1) 로 나온다)
y_test = y_test.ravel()

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

model.add(Dense(OUTPUT_DIM, activation='softmax'))  # 다중분류

model.summary()

# 3. 컴파일 훈련 (val_loss 가 PATIENCE 에폭 동안 좋아지지 않으면 멈춘다)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 -> 라벨
acc = accuracy_score(y_test, y_predict)  # y_test 는 이미 숫자 라벨이다

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
    f'|| acc={acc:.4f} '
    f'| stop={es.stopped_epoch if es.stopped_epoch else "ES미발동"} '
    f'| time={took:.1f}s | {took / (es.stopped_epoch or EPOCHS) * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
"""
