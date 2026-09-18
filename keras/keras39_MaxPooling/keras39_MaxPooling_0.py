# import np as np
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D
from tensorflow.keras.models import Sequential

#! 2. 모델구성
model = Sequential()
model.add(Input((10, 10, 1)))
model.add(
    Conv2D(
        10,
        (2, 2),  # (9,9,10)
        strides=1,  # 기본값
        padding='valid',  # 기본값
    )
)
model.add(MaxPool2D())
model.add(
    Conv2D(
        filters=9,
        kernel_size=(3, 3),  # (7,7,9)
        strides=2,
        padding='valid',
    )
)
model.summary()
