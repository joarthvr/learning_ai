import time

from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import boston_housing
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

# --- 훈련 ---
# GPU/CPU 시간 비교가 목적이므로 EarlyStopping 을 쓰지 않는다.
# ES 를 쓰면 멈추는 에폭이 매번 달라져서 걸린 시간을 비교할 수 없다.
EPOCHS = 100
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

# ======================================================================
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # fit 은 train 에만 (데이터 누수 방지)
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
| subject=boston | device=CPU | seed=42 | units=64-32-16-8 | act=relu | do=0.2-0.2-0.2-0.2 | ts=0.8 | vs=0.15 | bs=4 | ep=100 | lr=0.001
|| rmse=4.2107 | time=19.7s | 196.8ms/epoch |
===================================
===== RESULT =====
| subject=boston | device=GPU | seed=42 | units=64-32-16-8 | act=relu | do=0.2-0.2-0.2-0.2 | ts=0.8 | vs=0.15 | bs=4 | ep=100 | lr=0.001
|| rmse=3.6009 | time=28.4s | 283.7ms/epoch |
===================================

| subject=boston | seed=42 | units=64-32-16-8 | act=relu | do=0.2-0.2-0.2-0.2 | ts=0.8 | vs=0.15 | bs=4 | ep=100 | lr=0.001
|| rmse=6.6291 | time=34.8s | 348.5ms/epoch |

"""
