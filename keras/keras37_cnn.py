from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Input(shape=(5, 5, 1)))
# 서로 다른 2×2 커널 10개가 입력 전체를 각각 스캔해서, 특징 맵 10개를 만든다.
model.add(Conv2D(10, (2, 2)))  # Conv2D(필터_개수, 커널_크기, 입력 값의 크기)
model.add(Conv2D(5, (2, 2)))

model.summary()
