import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
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
]
ACTIVATION = 'relu'  # 층마다 다르게 주려면 LAYERS 에 'activation' 키를 넣으면 된다

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
SUBJECT = 'santander'
# ======================================================================

# 1. 데이터
DATA_PATH = './_data/kaggle_santander/'

train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
submission = pd.read_csv(DATA_PATH + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

INPUT_DIM = x.shape[1]  # 200
OUTPUT_DIM = 1  # 이진분류

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


# ======================================================================
# 2. 모델 구성 (함수형 API)
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
    out_activation='sigmoid',  # 이진분류
)
model.summary()
# ======================================================================

# 3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
y_prob = model.predict(x_test)
y_predict = np.round(y_prob)  # 0.5 기준 -> 0 또는 1
acc = accuracy_score(y_test, y_predict)

# target 이 90:10 이라 acc 는 전부 0으로 찍어도 0.90 -> AUC 를 같이 볼 것
auc = roc_auc_score(y_test, y_prob)

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
    f'|| acc={acc:.4f} | auc={auc:.4f} '
    f'| stop={stop_ep} | time={took:.1f}s |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================

"""
