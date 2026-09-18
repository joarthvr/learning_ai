import time

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
)
from tensorflow.keras.models import Model

SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
VAL_SPLIT = 0.2
TARGET_SIZE = 200
IMAGE_BATCH_SIZE = 10000
STEP = 'cat&dog'

#! 1. 데이터 -----------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,  # * 수평 뒤집기 (고양이/개는 좌우 대칭이라 안전)
    # vertical_flip=True,  # * 수직 뒤집기 — 뒤집힌 동물 사진은 없으므로 끔
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,  # * 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.15,
    shear_range=0.2,  # * 좌표하나를 고정하고 다른 몇개의 좌표를 이동(찌부시킴)
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

path_train = './_data/image/cat_dog/training_set/'
path_test = './_data/image/cat_dog/test_set/'

xy_train = train_datagen.flow_from_directory(
    path_train,  # 경로
    target_size=(TARGET_SIZE, TARGET_SIZE),  # * 동적으로 이미지에 맞게 조절됨
    batch_size=IMAGE_BATCH_SIZE,
    class_mode='binary',  # * 이진 분류
    color_mode='rgb',  # * 흑백
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,  # 경로
    target_size=(TARGET_SIZE, TARGET_SIZE),  # * 동적으로 이미지에 맞게 조절됨
    batch_size=IMAGE_BATCH_SIZE,
    class_mode='binary',  # * 이진 분류
    color_mode='rgb',  # * 흑백
    shuffle=False,  # test에서는 할 필요가 없음
)
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)  # ? (8000, 200, 200, 3) (8000,)
print(x_test.shape, y_test.shape)  # ? (2000, 200, 200, 3) (2000,)

#! 2. 모델 구성 -----------------------------------------------------------
inputs = Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D()(x)  # 200 -> 100
x = Dropout(0.2)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D()(x)  # 100 -> 50
x = Dropout(0.2)(x)

# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = MaxPool2D()(x)  # 50 -> 25
# x = Dropout(0.2)(x)

x = GlobalAveragePooling2D()(x)  # 25x25x128 -> 128
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
#! 3. 컴파일, 훈련 -----------------------------------------------------------

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
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
loss_value = loss[0] if isinstance(loss, (list, tuple)) else loss
y_predict = np.round(model.predict(x_test))  # 0.5 기준 -> 0 또는 1
acc = accuracy_score(y_test, y_predict)  # y_test 는 원핫이 아니라 0/1 라벨

print('')
print('===== RESULT =====')
print(f'|| acc={acc:.4f} | loss={loss_value:.4f} | time={end_time - start_time:.1f}s |')
print('===================================')
print('')
"""
======================= 실험 기록 =======================
===== RESULT ===== 0.77
|| acc=0.8324 | loss=0.4080 | time=254.2s |
|| acc=0.7820 | loss=0.4951 | time=598.6s |
===================================
"""
