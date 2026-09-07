from sklearn.datasets import fetch_california_housing, load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.9,       # 디폴트 75%
    test_size=0.1,        # 테스트 30%
    random_state=153,      # 재현성 보장
    shuffle=True          # 섞기 (기본값)
)

#2. 모델 설정
model = Sequential()
model.add(Dense(3, input_dim=10))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)


#3. 결과 예측
loss = model.evaluate(x_test, y_test)

print(loss)
# 2382.875244140625
# 2384.713134765625
# 2421.321533203125
# 2471.36669921875
# 2507.222900390625
# 2543.166748046875
# 2552.897216796875
# 2562.499755859375
# 2593.18798828125
# 2586.5703125
# 2634.714599609375
# 2705.389892578125
# 2808.1552734375
# 2816.995361328125
# 2899.655517578125
