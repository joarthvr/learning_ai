import time

from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense, Input
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

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 4
LEARNING_RATE = 0.001
PATIENCE = 20
# ======================================================================

# 스케일링 유무만 바꿔가며 두 번 돌린다. 나머지 조건은 전부 동일해야
# rmse 차이를 '스케일링 효과'라고 말할 수 있다 (한 번에 하나만 바꾸기)
CASES = [
    ('스케일링X', None),
    ('RobustScaler', RobustScaler()),
]

results = []

for case_name, scaler in CASES:
    # 매 실험 시작마다 난수를 다시 고정 -> 가중치 초깃값까지 두 실험이 완전히 같아짐
    set_random_seed(SEED)

    print('')
    print(f'########## [{case_name}] 훈련 시작 ##########')

    # 1. 데이터
    # boston_housing 은 이미 train/test 로 나뉘어 나오므로 train_test_split 이 없다
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(
        test_split=1 - TRAIN_SIZE,
        seed=SEED,
    )

    INPUT_DIM = x_train.shape[1]  # 13
    OUTPUT_DIM = 1  # 회귀: 숫자 1개

    # ======================================================================
    # scaler 가 None 인 회차는 이 블록을 통째로 건너뛴다 = 원본 그대로 학습
    if scaler is not None:
        # fit 은 x_train 에만! x_test 는 transform 만 (데이터 누수 방지)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    # ======================================================================

    # 2. 모델 구성
    model = Sequential()
    model.add(Input(shape=(INPUT_DIM,)))
    for units in HIDDEN_UNITS:
        model.add(Dense(units, activation=ACTIVATION))
    model.add(Dense(OUTPUT_DIM))

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

    results.append(
        {
            'name': case_name,
            'rmse': rmse,
            'stop': es.stopped_epoch if es.stopped_epoch else 'ES미발동',
            'time': end_time - start_time,
        }
    )

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(map(str, HIDDEN_UNITS))

print('')
print('===== RESULT =====')
for r in results:
    print(
        f'| scaler={r["name"]:<12} | seed={SEED} | units={structure} | act={ACTIVATION} '
        f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
        f'| lr={LEARNING_RATE} | pat={PATIENCE} '
        f'|| rmse={r["rmse"]:.4f} '
        f'| stop={r["stop"]} | time={r["time"]:.1f}s |'
    )

before, after = results[0]['rmse'], results[1]['rmse']
print(f'-> rmse {before:.4f} -> {after:.4f}  ({(before - after) / before * 100:+.1f}%)')
print('===================================')
print('')


"""

===== RESULT ===== // 민맥스
| scaler=스케일링X        | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=8 |ep=1000 | lr=0.001 | pat=20
|| rmse=4.2200 | stop=143 | time=11.8s |
| scaler=MinMaxScaler | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=8 | ep=1000 | lr=0.001 | pat=20
|| rmse=3.1154 | stop=489 | time=37.8s |
-> rmse 4.2200 -> 3.1154  (+26.2%)
===================================
===== RESULT ===== // 스탠다드
| scaler=스케일링X        | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=8 | ep=1000 | lr=0.001 | pat=20 || rmse=4.2200 | stop=143 | time=11.8s |
| scaler=MinMaxScaler | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=8 | ep=1000 | lr=0.001 | pat=20 || rmse=3.9134 | stop=36 | time=3.5s |
-> rmse 4.2200 -> 3.9134  (+7.3%)
===================================
===== RESULT ===== // abs
| scaler=스케일링X        | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=8 | ep=1000 | lr=0.001 | pat=20 || rmse=4.2200 | stop=143 | time=11.5s |
| scaler=MinMaxScaler | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=8 | ep=1000 | lr=0.001 | pat=20 || rmse=3.5218 | stop=339 | time=27.0s |
-> rmse 4.2200 -> 3.5218  (+16.5%)
===================================
===== RESULT ===== // abs
| scaler=스케일링X        | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=16 | ep=1000 | lr=0.001 | pat=20 || rmse=4.5315 | stop=184 | time=11.5s |
| scaler=MinMaxScaler | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=16 | ep=1000 | lr=0.001 | pat=20 || rmse=3.3098 | stop=688 | time=41.7s |
-> rmse 4.5315 -> 3.3098  (+27.0%)
===================================
===== RESULT ===== // abs
| scaler=스케일링X        | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=32 | ep=1000 | lr=0.001 | pat=20 || rmse=5.4964 | stop=146 | time=8.3s |
| scaler=MaxAbsScaler | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=32 | ep=1000 | lr=0.001 | pat=20 || rmse=3.7341 | stop=337 | time=18.1s |
-> rmse 5.4964 -> 3.7341  (+32.1%)
===================================
===== RESULT ===== robust
| scaler=스케일링X        | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=4 | ep=1000 | lr=0.001 | pat=20 || rmse=4.3209 | stop=124 | time=14.0s |
| scaler=RobustScaler | seed=42 | units=64-32-16-8 | act=relu | ts=0.8 | vs=0.15 | bs=4 | ep=1000 | lr=0.001 | pat=20 || rmse=3.5447 | stop=44 | time=5.5s |
-> rmse 4.3209 -> 3.5447  (+18.0%)
===================================
"""
