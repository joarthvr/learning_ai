import time

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
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
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
SUBJECT = 'bike'
# ======================================================================

# 1. 데이터
DATA_PATH = './_data/kaggle_bike/'

train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
submission = pd.read_csv(DATA_PATH + 'sampleSubmission.csv', index_col=0)

# casual + registered = count 이므로 남겨두면 답을 보고 푸는 셈 (타깃 누수)
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

INPUT_DIM = x.shape[1]  # 8
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


# ======================================================================
# 2. 모델 구성 (함수형 API) — 여기부터 직접 작성
#
#   1) inputs = Input(shape=(input_dim,))
#   2) x = inputs 로 두고, layers 를 돌며 Dense / Dropout 을 x 에 연결
#        x = Dense(...)(x)   <- 레이어를 '함수처럼 호출'해서 앞 층의 출력을 넘긴다
#      dropout 은 layer.get('dropout', 0) 으로 꺼내고 0 보다 클 때만 붙인다
#   3) outputs = Dense(output_dim, activation=out_activation)(x)
#   4) return Model(inputs=inputs, outputs=outputs)
#
#   참고: keras34_functional_02_diabetes.py
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
======================= 실험 기록 =======================

"""
