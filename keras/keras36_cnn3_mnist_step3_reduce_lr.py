"""3단계: 2단계 + ReduceLROnPlateau.

2단계 대비 바뀐 점
  - val_loss 가 정체되면 학습률을 절반으로 낮춘다.
    큰 보폭으로는 못 들어가는 좁은 골짜기를 보폭을 줄여 파고드는 것.
    0.99 후반대를 뚫을 때 효과가 크다.
  - ES 의 patience 를 RLR 보다 넉넉히 준다 (LR 을 낮춘 뒤 개선될 시간을 줘야 함)
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
VAL_SPLIT = 0.2
STEP = 'step3_reduce_lr'
PATIENCE = 15  # RLR(5) 보다 넉넉하게
RLR_PATIENCE = 5
# ======================================================================

set_random_seed(SEED)

#! 1. 데이터 -----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#! 스케일링 - 방법 하나만 쓴다 (겹쳐 쓰면 범위가 망가진다)
# 픽셀은 0~255 고정 상수라 train 기준으로 fit 할 필요가 없다 -> 데이터 누수 아님
x_train = x_train / 255.0
x_test = x_test / 255.0
print('스케일링 후 범위:', np.max(x_train), np.min(x_train))

#! CNN 은 펼치지 않고 (28, 28, 1) 형태로 넣는다. 채널 축만 추가
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)

#! 원핫 인코더 -----------------------------------------------------------
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))  # test 는 transform 만!
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#! 2. 모델 구성 -----------------------------------------------------------
# 필터는 뒤로 갈수록 늘리고(32->64), 크기는 풀링으로 줄인다.
# 앞쪽은 선/모서리 같은 단순한 패턴이라 종류가 적고, 뒤로 갈수록 조합이 많아진다.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))  # (26, 26, 32)
model.add(MaxPooling2D((2, 2)))  # (13, 13, 32)
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu'))  # (11, 11, 64)
model.add(MaxPooling2D((2, 2)))  # (5, 5, 64)
model.add(Dropout(0.25))

model.add(Flatten())  # 5*5*64 = 1600
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dense 앞은 세게
model.add(Dense(10, activation='softmax'))

model.summary()

#! 3. 컴파일, 훈련 -----------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',  # 정확도는 높을수록 좋으므로 max
    patience=PATIENCE,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.5,  # 정체되면 학습률 절반
    patience=RLR_PATIENCE,
    min_lr=1e-6,
    verbose=1,
)

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=VAL_SPLIT,
    callbacks=[es, rlr],
)
end_time = time.time()

#! 4. 평가, 예측 -----------------------------------------------------------
loss = model.evaluate(x_test, y_test, verbose=0)

y_predict = np.argmax(model.predict(x_test), axis=1)
y_test_label = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_label, y_predict)

print('')
print('===== RESULT =====')
print(
    f'| step={STEP} | seed={SEED} | bs={BATCH_SIZE} | ep={EPOCHS} | pat={PATIENCE} '
    f'|| acc={acc:.4f} | loss={loss[0]:.4f} '
    f'| stop={es.stopped_epoch if es.stopped_epoch else "ES미발동"} '
    f'| time={end_time - start_time:.1f}s |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================
목표: acc > 0.995
===== RESULT =====
#? step=step3_reduce_lr | seed=42 | bs=128 | ep=50 | pat=15 || acc=0.9960 | loss=0.0193 | stop=48 | time=191.7s |
===================================

"""
