import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---
LAYERS = [
    {'units': 64, 'dropout': 0.2},
    {'units': 32, 'dropout': 0.2},
    {'units': 16, 'dropout': 0.2},
]
ACTIVATION = 'relu'

# --- 훈련 ---
# GPU/CPU 시간 비교가 목적이므로 EarlyStopping 을 쓰지 않는다.
# ES 를 쓰면 멈추는 에폭이 매번 달라져서 걸린 시간을 비교할 수 없다.
EPOCHS = 100
BATCH_SIZE = 72
LEARNING_RATE = 0.001

set_random_seed(SEED)
SUBJECT = 'digits'
# ======================================================================

##################### GPU 사용 가능 여부 확인 #####################
gpus = tf.config.list_physical_devices('GPU')
DEVICE = 'GPU' if gpus else 'CPU'
print('')
print(f'TensorFlow {tf.__version__} / 사용 장치: {DEVICE}')
if gpus:
    print('GPU 목록:', gpus)
else:
    # TF 2.11 부터 윈도우 네이티브는 GPU 를 지원하지 않는다 (WSL2 필요)
    print('GPU 없음 -> CPU 로 실행')
print('')
##############################################################################

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = pd.get_dummies(datasets.target, dtype='float32')  # 원핫 (1797, 10)

INPUT_DIM = x.shape[1]  # 64
OUTPUT_DIM = y.shape[1]  # 10

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y,  # 클래스 비율 유지
)

# ======================================================================
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # fit 은 train 에만 (데이터 누수 방지)
# ======================================================================


# 2. 모델 구성 (함수형 API)
def build_model(input_dim, output_dim, layers, activation, out_activation=None):
    inputs = Input(shape=(input_dim,))

    x = inputs
    for layer in layers:
        x = Dense(layer['units'], activation=layer.get('activation', activation))(x)
        dropout = layer.get('dropout', 0)
        if dropout > 0:
            x = Dropout(dropout)(x)

    outputs = Dense(output_dim, activation=out_activation)(x)
    return Model(inputs=inputs, outputs=outputs)


model = build_model(
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    layers=LAYERS,
    activation=ACTIVATION,
    out_activation='softmax',  # 다중분류
)
model.summary()

# 3. 컴파일 훈련 (EarlyStopping 없이 EPOCHS 를 끝까지 돌린다)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()
hist = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=VAL_SPLIT,
)
end_time = time.time()

# 4. 결과 예측
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 -> 라벨
y_test_label = np.argmax(y_test, axis=-1)  # 원핫 -> 라벨
acc = accuracy_score(y_test_label, y_predict)

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(str(layer['units']) for layer in LAYERS)
dropouts = '-'.join(str(layer.get('dropout', 0)) for layer in LAYERS)
took = end_time - start_time

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT} | device={DEVICE} | seed={SEED} '
    f'| units={structure} | act={ACTIVATION} | do={dropouts} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} '
    f'|| acc={acc:.4f} '
    f'| time={took:.1f}s | {took / EPOCHS * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
===== RESULT ===== cpu
| subject=digits | device=CPU | seed=42 | units=64-32-16 | act=relu | do=0.2-0.2-0.2 | ts=0.8 | vs=0.15 | bs=72 | ep=100 | lr=0.001
|| acc=0.9722 |time=10.9s | 108.9ms/epoch |
===================================

===== RESULT ===== gpu
| subject=digits | device=GPU | seed=42 | units=64-32-16 | act=relu | do=0.2-0.2-0.2 | ts=0.8 | vs=0.15 | bs=72 | ep=100 | lr=0.001
|| acc=0.9667 | time=10.1s | 101.0ms/epoch |
===================================
"""
