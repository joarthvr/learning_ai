import time

from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Input, Model
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---
# 한 층의 설정을 한 줄에 모은다. 리스트 길이 = 은닉층 수
LAYERS = [
    {'units': 64, 'dropout': 0.2},
    {'units': 32, 'dropout': 0.2},
    {'units': 16, 'dropout': 0.2},
    {'units': 8, 'dropout': 0.2},
]
ACTIVATION = 'relu'  # 층마다 다르게 주려면 LAYERS 에 'activation' 키를 넣으면 된다

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 4
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
SUBJECT = 'boston'
# ======================================================================

# 1. 데이터
# boston_housing 은 이미 train/test 로 나뉘어 나오므로 train_test_split 이 없다
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(
    test_split=1 - TRAIN_SIZE,
    seed=SEED,
)

INPUT_DIM = x_train.shape[1]  # 13
OUTPUT_DIM = 1  # 회귀

# ======================================================================
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # fit 은 train 에만 (데이터 누수 방지)
# ======================================================================


# ======================================================================
# 2. 모델 구성 (함수형 API) — 여기부터 직접 작성
def build_model(input_dim, output_dim, layers, activation, out_activation=None):
    inputs = Input(shape=(input_dim,))

    # 함수형의 핵심: x = 레이어(x) 로 계속 덮어써야 사슬이 이어진다.
    # 'x =' 를 빠뜨리면 그 레이어는 모델에 연결되지 않고 조용히 사라진다.
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
)
model.summary()
# ======================================================================

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
structure = '-'.join(str(layer['units']) for layer in LAYERS)
dropouts = '-'.join(str(layer.get('dropout', 0)) for layer in LAYERS)
stop_ep = es.stopped_epoch if es.stopped_epoch else 'ES미발동'
took = end_time - start_time

print('')
print('===== RESULT =====')
print(
    f'| subject={SUBJECT} | seed={SEED} | units={structure} | act={ACTIVATION} '
    f'| do={dropouts} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| rmse={rmse:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')


"""
"""
