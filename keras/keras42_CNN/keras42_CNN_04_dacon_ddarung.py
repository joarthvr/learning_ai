import time

import pandas as pd
from sklearn.metrics import root_mean_squared_error
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
    {'units': 8, 'dropout': 0.2},
]
ACTIVATION = 'relu'
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
set_random_seed(SEED)
SUBJECT = 'ddarung'
# ======================================================================


# 1. 데이터
DATA_PATH = './_data/ddarung/'

train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
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
model.add(GlobalAveragePooling2D())
model.add(Dense(OUTPUT_DIM, activation='relu'))

# 3. 컴파일 훈련 (EarlyStopping 없이 EPOCHS 를 끝까지 돌린다)
model.compile(loss='mse', optimizer='adam')

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
rmse = root_mean_squared_error(y_test, model.predict(x_test))

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
    f'|| rmse={rmse:.4f} '
    f'| time={took:.1f}s | {took / EPOCHS * 1000:.1f}ms/epoch |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
===== RESULT ===== cpu
| subject=ddarung | device=CPU | seed=42 | units=64-32-16-8 | act=relu | do=0.2-0.2-0.2-0.2 | ts=0.8 | vs=0.15| bs=8 | ep=100 | lr=0.001
|| rmse=45.7292 | time=28.6s | 285.7ms/epoch |
===================================
===== RESULT ===== gpu
| subject=ddarung | device=GPU | seed=42 | units=64-32-16-8 | act=relu | do=0.2-0.2-0.2-0.2 | ts=0.8 | vs=0.15 | bs=8 | ep=100 | lr=0.001
|| rmse=51.2237 | time=35.8s | 358.4ms/epoch |
===================================

| subject=ddarung | seed=42 | units=64-32-16-8 | act=relu | do=0.2-0.2-0.2-0.2 |ts=0.8 | vs=0.15 | bs=8 | ep=100 | lr=0.001
|| rmse=55.1051 | time=46.7s | 466.7ms/epoch |
"""
