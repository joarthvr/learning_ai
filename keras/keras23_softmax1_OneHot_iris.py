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
from tensorflow.keras.utils import set_random_seed, to_categorical

# ======================================================================
SEED = 42  # 난수 고정 (재현성) — 튜닝 대상 아님!

# --- 데이터 ---
TRAIN_SIZE = 0.7  # train / test 분할 비율
VAL_SPLIT = 0.2  # train 중 검증에 쓸 비율

# --- 모델 ---
HIDDEN_UNITS = [10, 10, 10]  # 은닉층 구조 (리스트 길이 = 층 수)
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.001  # Adam 기본값
PATIENCE = 20
# ======================================================================

set_random_seed(SEED)  # python / numpy / tensorflow 난수를 한 번에 고정


# 1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (150, 4) (150,)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50]))

############################ 원핫 인코딩 1 keras ############################
# 0 -> [1, 0, 0]  /  1 -> [0, 1, 0]  /  2 -> [0, 0, 1]
# softmax(출력 3개) + categorical_crossentropy 는 정답도 (n, 3) 이어야 함
y = to_categorical(y)
# print(y.shape)   # (150, 3)
# print(y)         # 변환 결과 눈으로 확인
############################################################################

############################ 원핫 인코딩 2 pandas ############################
# 0 -> [1, 0, 0]  /  1 -> [0, 1, 0]  /  2 -> [0, 0, 1]
# 주의: 기본 dtype이 bool 이므로 dtype 지정 필요. 결과는 DataFrame.
y = pd.get_dummies(datasets.target, dtype='float32')
# print(y)
############################################################################

############################ 원핫 인코딩 3 sklearn ############################
# 0 -> [1, 0, 0]  /  1 -> [0, 1, 0]  /  2 -> [0, 0, 1]
# 주의: 2차원 입력만 받으므로 reshape(-1, 1) 필요.
# 유일하게 fit/transform 이 분리되어 train 기준을 test 에 그대로 적용할 수 있다.
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(datasets.target.reshape(-1, 1))
# print(y.shape)   # (150, 3)
# print(y)
##############################################################################

# 세 방법의 결과가 모두 같은지 검산
# print(np.array_equal(to_categorical(datasets.target), y))   # True

# exit()   # 데이터 확인만 하고 멈출 때 주석 해제

# 데이터에서 유도되는 값 — 하이퍼파라미터가 아니므로 위 블록에 두지 않는다
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

# print(x_train.shape, x_test.shape) # (105, 4) (45, 4)
# print(y_train.shape, y_test.shape) # (105, 3) (45, 3)

# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(INPUT_DIM,)))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation=ACTIVATION))
#################################################
model.add(Dense(N_CLASSES, activation='softmax'))
#################################################

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
    validation_split=VAL_SPLIT,  # val_loss가 있어야 EarlyStopping이 작동함
    callbacks=[es],
)
end_time = time.time()

# 4. 평가 예측
#################################################
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 (45,3) -> 라벨 (45,)
y_test_label = np.argmax(y_test, axis=-1)  # 원핫 정답 -> 라벨
#################################################

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


# 모델은 학습을 위해 확률이라는 부드러운 언어로 말하도록 만들어졌습니다. argmax는 미분이 안 되므로 모델 바깥, 학습이 다 끝난 뒤에만 쓸 수 있음
# to_categorical은 훈련을 위해 정수를 원핫으로 바꾸고,
# argmax는 채점을 위해 원핫과 확률을 다시 정수로 내립니다.

# 둘 다 정상 흐름이고, 중간에 깨지는 일은 없습니다.

# 원핫으로 바꾸는 것: 정답 y를. 모델 출력이 3칸이라 정답도 3칸이어야 loss를 계산할 수 있어서.

# 되돌리는 것1: 정답 y_test를. sklearn이 정수만 읽어서. (형식 문제)

# 되돌리는 것2: 예측 확률을. 확률은 답이 아니라 골라야 해서. (판단 문제)

# 되돌리지 않는 것: y_train과 evaluate용 y_test. 받는 쪽(keras)이 원핫을 이해해서.


"""
======================= 실험 기록 =======================
RESULT 한 줄을 그대로 복사해서 아래에 붙여넣기 (설정이 줄 안에 다 들어있음)

1)

"""
