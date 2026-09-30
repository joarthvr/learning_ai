import time

import numpy as np
from keras.preprocessing.image import (
    ImageDataGenerator,  # tensorflow 빼도 됌 오히려 버전 2는 이게 편함
)
from sklearn.model_selection import train_test_split

PATH = 'C:/study/_data/image/men_women/'

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

read_time = time.time()
xy = datagen.flow_from_directory(
    PATH,
    (100, 100),
    color_mode='rgb',
    class_mode='binary',
    batch_size=30000,
    shuffle=False,
)

x = xy[0][0]
x = xy[0][0]
y = xy[0][1]
print(
    np.unique(y, return_counts=True)
)  # (array([0., 1.], dtype=float32), array([17678,  9489], dtype=int64))
print(x.shape, y.shape)  # (27167, 100, 100, 3) (27167,)
read_end_time = time.time()

print(f'읽기 시간 = {read_end_time - read_time:.4f}')

# x_train의 여자 데이터만 추출하여 증강 진행
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)

print(x_train.shape, x_test.shape)  # (20375, 100, 100, 3) (6792, 100, 100, 3)
print(y_train.shape, y_test.shape)  # (20375,) (6792,)

female_idx = np.where(y_train == 1)
x_female = x_train[female_idx].copy()
y_female = y_train[female_idx].copy()

print(len(x_female))

aug_start = time.time()
datagen = ImageDataGenerator(
    rotation_range=20, horizontal_flip=True, width_shift_range=0.1, fill_mode='nearest'
)

augment_size = 8000

idx = np.random.choice(x_female.shape[0], size=augment_size)

x_aug = x_female[idx].copy()
y_aug = y_female[idx].copy()

print(x_aug.shape, y_aug.shape)

x_aug = datagen.flow(x_aug, y_aug, batch_size=augment_size, shuffle=False).next()[0]

# 합친 뒤 뒤에 8천개가 여자임으로 다시 데이터를 섞어줌
x_train = np.concatenate((x_train, x_aug))
y_train = np.concatenate((y_train, y_aug))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

aug_end = time.time()

print(x_train.shape, x_test.shape)  # (28375, 100, 100, 3) (6792, 100, 100, 3)
print(y_train.shape, y_test.shape)  # (28375,) (6792,)

np_path = '_save/keras51/gender/'
np.save(np_path + 'keras51_gender_x_train.npy', arr=x_train)
np.save(np_path + 'keras51_gender_x_test.npy', arr=x_test)
np.save(np_path + 'keras51_gender_y_train.npy', arr=y_train)
np.save(np_path + 'keras51_gender_y_test.npy', arr=y_test)

print('Save Completed')
print(f'증강/저장 시간 = {aug_end - aug_start:.4f}')
