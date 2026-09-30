# [학습 정리] keras46 가위바위보(rps) → npy 저장 (짝: keras47_load_npy_02_rps)
# - class_mode 를 안 주면 기본값 'categorical' → y 가 이미 원핫 (N, 3) 으로 나온다.
#   그래서 01 처럼 OneHotEncoder 를 따로 쓸 필요가 없다.

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

PATH = './_data/image/rps/'

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

xy = datagen.flow_from_directory(
    PATH,
    (200, 200),
    color_mode='rgb',
    batch_size=2050,
    shuffle=True,
)

x = xy[0][0]
y = xy[0][1]

print(x.shape, y.shape)  # (2048, 200, 200, 3) (2048, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y)

print(x_train.shape, x_test.shape)  # (1536, 200, 200, 3) (512, 200, 200, 3)
print(y_train.shape, y_test.shape)  # (1536, 3) (512, 3)

np_path = '_save/keras46/rps/'
np.save(np_path + 'keras46_rps_x_train.npy', arr=x_train)
np.save(np_path + 'keras46_rps_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_rps_y_train.npy', arr=y_train)
np.save(np_path + 'keras46_rps_y_test.npy', arr=y_test)

print('Save Completed')
