import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(np.__version__)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,  # * 수평 뒤집기
    vertical_flip=True,  # * 수직 뒤집기
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,  # * 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=1.2,
    shear_range=0.7,  # * 좌표하나를 고정하고 다른 몇개의 좌표를 이동(찌부시킴)
    fill_mode='nearest',
)

#! 평가 훈련할 데이터는 변환할 필요가 없음
#! 테스트해야하는 데이터이기 때문
test_datagen = ImageDataGenerator(
    rescale=1.255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,  # 경로
    target_size=(200, 200),  # * 동적으로 이미지에 맞게 조절됨
    batch_size=10,
    class_mode='binary',  # * 이진 분류
    color_mode='grayscale',  # * 흑백
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,  # 경로
    target_size=(200, 200),  # * 동적으로 이미지에 맞게 조절됨
    batch_size=10,
    class_mode='binary',  # * 이진 분류
    color_mode='grayscale',  # * 흑백
    huffle=False,  # test에서느 할 필요가 없음
)

print(len(xy_train))  # ? 16
print(xy_train[0][0].shape)  # ? (10, 200, 200, 1)
print(xy_train[0][1].shape)  # ? (10,)
print(type(xy_train))  # ? <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  # ? <class 'tuple'>
print(type(xy_train[0][0]))  # ? <class 'numpy.ndarray'>
print(type(xy_train[0][1]))  # ? <class 'numpy.ndarray'>
