"""
cat_dog 이미지를 npy 로 저장한다 (짝: keras45_04_catdog_load_npy.py).
flow_from_directory 는 매번 수천 장을 디스크에서 읽고 리사이즈해서 느리다.
한 번 numpy 배열로 굳혀두면 load 쪽에서 즉시 꺼내 쓸 수 있다.
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# ======================================================================
# --- 데이터 ---
IMG_SIZE = (180, 180)  # 200 은 8005장이면 3.8GB 라 OOM 이 난다
COLOR_MODE = 'rgb'  # * 컬러
TRAIN_SIZE = 10000  # flow_from_directory 가 찾은 전체 장수
TEST_SIZE = 4000

PATH_TRAIN = './_data/image/cat_dog/training_set/'
PATH_TEST = './_data/image/cat_dog/test_set/'
PATH_NPY = './_data/kaggle_cat_dog_npy/'
PREFIX = 'keras45_03_'

SUBJECT = 'cat_dog'
# ======================================================================

#! 1. 데이터 -----------------------------------------------------------
#! 저장 단계에서는 증강을 걸지 않는다.
#! 증강은 에폭마다 달라져야 의미가 있는데, npy 로 굳히면 한 장으로 고정된다.
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

xy_train = train_datagen.flow_from_directory(
    PATH_TRAIN,
    target_size=IMG_SIZE,
    batch_size=TRAIN_SIZE,  # 한 번에 전부 꺼낸다
    class_mode='binary',  # * 이진 분류
    color_mode=COLOR_MODE,
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    PATH_TEST,
    target_size=IMG_SIZE,
    batch_size=TEST_SIZE,
    class_mode='binary',
    color_mode=COLOR_MODE,
    shuffle=False,  # test 는 섞을 필요가 없음
)

x_train, y_train = xy_train[0]
x_test, y_test = xy_test[0]

print(x_train.shape, y_train.shape)  # ? (8005, 100, 100, 3) (8005,)
print(x_test.shape, y_test.shape)  # ? (2023, 100, 100, 3) (2023,)

#! 2. 저장 -----------------------------------------------------------
np.save(PATH_NPY + PREFIX + 'x_train.npy', arr=x_train)
np.save(PATH_NPY + PREFIX + 'y_train.npy', arr=y_train)
np.save(PATH_NPY + PREFIX + 'x_test.npy', arr=x_test)
np.save(PATH_NPY + PREFIX + 'y_test.npy', arr=y_test)

print('')
print('===== SAVED =====')
print(f'| path={PATH_NPY} | prefix={PREFIX} ')
print(f'| img={IMG_SIZE[0]}x{IMG_SIZE[1]} | color={COLOR_MODE} ')
print(f'| train={x_train.shape} | test={x_test.shape} |')
print('===================================')
print('')
