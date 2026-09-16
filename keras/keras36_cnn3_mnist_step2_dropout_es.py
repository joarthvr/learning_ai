"""2단계: 1단계 + Dropout 위치 조정 + EarlyStopping.

1단계 대비 바뀐 점
  - Dropout: 풀링 뒤 0.25, Dense 앞 0.5 (파라미터가 몰린 Dense 쪽이 진짜 과적합 지점)
  - EarlyStopping(val_acc, max) 으로 최고 시점의 가중치를 복원
    -> 50 에폭을 끝까지 돌리면 최고점을 지나 과적합으로 내려간 상태가 남는다
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
VAL_SPLIT = 0.2
STEP = 'step2_dropout_es'
PATIENCE = 10
# ======================================================================

set_random_seed(SEED)

#! 1. 데이터 -----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#! 스케일링
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

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=VAL_SPLIT,
    callbacks=[es],
)
end_time = time.time()

#! 4. 평가, 예측 -----------------------------------------------------------
loss = model.evaluate(x_test, y_test, verbose=0)

y_predict = np.argmax(model.predict(x_test), axis=1)
y_test_label = np.argmax(y_test, axis=1)  # y_test 를 넣어야 한다 (y_predict 아님!)
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
===== RESULT =====
# ?  step=step2_dropout_es | seed=42 | bs=128 | ep=50 | pat=10 || acc=0.9953 | loss=0.0151 | stop=25 | time=423.4s |
===================================

"""
