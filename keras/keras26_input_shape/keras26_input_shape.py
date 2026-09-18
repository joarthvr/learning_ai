import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.7
VAL_SPLIT = 0.2

# --- 모델 ---
HIDDEN_UNITS = [10, 10, 10]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.001  # Adam 기본값
PATIENCE = 20

set_random_seed(SEED)  # python / numpy / tensorflow 난수를 한 번에 고정
# ======================================================================


# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
y = pd.get_dummies(datasets.target, dtype='float32')

INPUT_DIM = x.shape[1]  # 4
N_CLASSES = np.shape(y)[1]  # 3

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
# ======================================================================

model.add(Input(shape=(INPUT_DIM,)))
# model.add(Dense(10, input_shape=(INPUT_DIM,)))

# ======================================================================

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
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 (45,3) -> 라벨 (45,)
y_test_label = np.argmax(y_test, axis=-1)  # 원핫 정답 -> 라벨
acc = accuracy_score(y_test_label, y_predict)
train_acc = model.evaluate(x_train, y_train, verbose=0)[1]
test_acc = model.evaluate(x_test, y_test, verbose=0)[1]

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
    f'|| train={train_acc:.4f} | test={test_acc:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')


"""



"""
