# 36-2 카피
import numpy as np
from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)
print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

# ! 이미지 전처리

# ! 스케일링 1 Minmax
# * 스케일링은 x_train이 대상이다
# * 최소값이 0이기 떄문에 255로 나누는 거임 0~1
x_train = x_train / 255.0
x_test = x_test / 255.0
print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))


# !스케일링 2 Maxabs
# * -1.0 ~ 1.0
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))
