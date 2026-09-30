# 50-1 복사

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

################### 증폭 ###################
datagen = ImageDataGenerator(  # 이 중에 랜덤으로 몇개 적용
    rescale=1.0 / 255,  # . 수차화는 여기, 붙이면 형변환 됐단 표시 (float) = 개발자들의 룰
    horizontal_flip=True,  # 수평 뒤집기 (좌우반전)
    # vertical_flip=True, # 수집 뒤집기 (상하반전)
    width_shift_range=0.1,  # 평행 이동
    # height_shift_range=0.1,
    rotation_range=15,  # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.1,
    # shear_range=0.7, # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode='nearest',
)

augment_size = 100
print(x_train.shape)  # (60000, 28, 28)
print(x_train[0].shape)  # (28, 28)

aaa = np.tile(x_train[0], augment_size)  # 단순 복붙
print(aaa.shape)  # (28, 2800) (복제한게 가로, 컬럼으로 쭉 이어 붙은 형태)

aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)  # (100, 28, 28, 1)

xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28 * 28), augment_size).reshape(-1, 28, 28, 1),  # (100, 28, 28, 1)
    # reshape(28*28) 을 넣지 않으면 1개의 데이터가 온전히 증폭처리되지 않음.
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=False,
).next()

print(xy_data)
print(type(xy_data))  # tuple
# print(xy_data.shape)  # AttributeError: 'tuple' object has no attribute 'shape'
print(len(xy_data))  # 2 = x & y

print(xy_data[0].shape)  # (100, 28, 28, 1)
print(xy_data[1].shape)  # (100,)

plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i + 1)
    plt.imshow(xy_data[0][i], cmap='gray')
plt.grid(None)
plt.show()
