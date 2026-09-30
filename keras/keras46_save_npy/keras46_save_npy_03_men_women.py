# [학습 정리] keras46 남녀(men_women) → npy 저장 (짝: keras47_load_npy_03_men_women)
# - 2.7만 장이라 해상도를 100x100 으로 낮춤. 200x200 이면 메모리가 4배 → 약 13GB (float32).
# - 남:여 = 17678:9489 불균형 → stratify=y 로 train/test 비율을 똑같이 유지한다.
#   (이 불균형은 keras51_augment_05 에서 '여자만 증강'으로 다시 다룬다)

import numpy as np
from keras.preprocessing.image import (
    ImageDataGenerator,  # tensorflow 빼도 됌 오히려 버전 2는 이게 편함
)
from sklearn.model_selection import train_test_split

PATH = 'C:/study/_data/image/men_women/'

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

xy = datagen.flow_from_directory(
    PATH,
    (100, 100),
    color_mode='rgb',
    class_mode='binary',
    batch_size=30000,
    shuffle=False,  # 어차피 train_test_split 이 섞어준다
)

x = xy[0][0]
x = xy[0][0]
y = xy[0][1]
print(
    np.unique(y, return_counts=True)
)  # (array([0., 1.], dtype=float32), array([17678,  9489], dtype=int64))
print(x.shape, y.shape)  # (27167, 100, 100, 3) (27167,)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)

print(x_train.shape, x_test.shape)  # (20375, 100, 100, 3) (6792, 100, 100, 3)
print(y_train.shape, y_test.shape)  # (20375,) (6792,)

np_path = '_save/keras46/gender/'
np.save(np_path + 'keras46_gender_x_train.npy', arr=x_train)
np.save(np_path + 'keras46_gender_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_gender_y_train.npy', arr=y_train)
np.save(np_path + 'keras46_gender_y_test.npy', arr=y_test)

print('Save Completed')
