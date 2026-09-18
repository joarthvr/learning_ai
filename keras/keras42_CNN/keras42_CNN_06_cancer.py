import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, MaxPool2D
from tensorflow.keras.models import Sequential
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
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

set_random_seed(SEED)
SUBJECT = 'cancer'
# ======================================================================

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

INPUT_DIM = x.shape[1]
OUTPUT_DIM = 1
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y,
)

# ======================================================================
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(-1, INPUT_DIM, 1, 1)
x_test = x_test.reshape(-1, INPUT_DIM, 1, 1)
# ======================================================================


# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(INPUT_DIM, 1, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(GlobalAveragePooling2D())
model.add(Dense(OUTPUT_DIM, activation='sigmoid'))

# 3. 컴파일 훈련 (EarlyStopping 없이 EPOCHS 를 끝까지 돌린다)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
y_prob = model.predict(x_test)
y_predict = np.round(y_prob)  # 0.5 기준 -> 0 또는 1
acc = accuracy_score(y_test, y_predict)

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(str(layer['units']) for layer in LAYERS)
dropouts = '-'.join(str(layer.get('dropout', 0)) for layer in LAYERS)
took = end_time - start_time

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT} | seed={SEED} '
    f'| units={structure} | act={ACTIVATION} | do={dropouts} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} '
    f'|| acc={acc:.4f} '
    f'| time={took:.1f}s | {took / EPOCHS * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
===== RESULT ===== cpu
| subject=cancer | device=CPU | seed=42 | units=64-32-16 | act=relu | do=0.2-0.2-0.2 | ts=0.8 | vs=0.15 | bs=16 | ep=100 | lr=0.001
|| acc=0.9561 |time=11.3s | 113.0ms/epoch |
===================================
===== RESULT ===== gpu
| subject=cancer | device=GPU | seed=42 | units=64-32-16 | act=relu | do=0.2-0.2-0.2 | ts=0.8 | vs=0.15 | bs=16 | ep=100 | lr=0.001
|| acc=0.9561 | time=14.2s | 142.2ms/epoch |
===================================
| subject=cancer | seed=42 | units=64-32-16 | act=relu | do=0.2-0.2-0.2 | ts=0.8| vs=0.15 | bs=16 | ep=100 | lr=0.001
|| acc=0.9035 | time=20.9s | 208.7ms/epoch|
"""
