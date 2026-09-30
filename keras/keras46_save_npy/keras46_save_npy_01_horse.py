# [학습 정리] keras46 이미지 → npy 저장 (짝: keras47_load_npy_01_horse)
# - jpg 는 읽을 때마다 디코딩 + 리사이즈 비용이 든다. 한 번만 배열로 만들어 npy 로 저장해두면
#   다음부터는 np.load 한 줄로 끝 → 실험 반복 속도가 크게 빨라진다.
# - batch_size 를 전체 장수 이상으로 주면 xy[0] 한 배치에 데이터 전부가 들어온다.

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터

PATH = r'.\_data\image\horse-human'

datagen = ImageDataGenerator(rescale=1.0 / 255)

print(datagen)

xy = datagen.flow_from_directory(
    PATH,
    target_size=(200, 200),  # 꼭 이미지와 같은 해상도를 넣을 필요는 없음 알아서 바꿔줌.
    batch_size=160,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

x_train = xy[0][0]  # xy[배치번호][0=x, 1=y]
y_train = xy[0][1]

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
print(y_train.shape)

print(x_train.shape, y_train.shape)  # (160, 200, 200, 3) (160,)  # → OHE 후라 실제 y 는 (160, 2)
# [주의] [:100] / [101:] 이라 100번 샘플 하나가 빠진다 → test 59장
x_test = x_train[101:]
x_train = x_train[:100]
y_test = y_train[101:]
y_train = y_train[:100]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np_path = '_save/keras46/horse/'
np.save(np_path + 'keras46_horse_x_train.npy', arr=x_train)
np.save(np_path + 'keras46_horse_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_horse_y_train.npy', arr=y_train)
np.save(np_path + 'keras46_horse_y_test.npy', arr=y_test)

print('Save Completed')
