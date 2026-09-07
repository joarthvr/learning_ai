from sklearn.datasets import fetch_california_housing
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,       # 디폴트 75%
    test_size=0.3,        # 테스트 30%
    random_state=42,      # 재현성 보장
    shuffle=True          # 섞기 (기본값)
)

#2. 모델 설정
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

#3. 결과 예측
loss = model.evaluate(x_test, y_test)

print(loss)
# 0.6144575476646423
# 0.6213623285293579
# 0.674436092376709
# 0.6243575215339661
# 0.6240781545639038
# 0.6227288842201233
# 0.6334179639816284
# 0.6483346223831177
# 0.6615400910377502
# 0.6623608469963074
# 0.6643620133399963
# 0.6413103342056274