# 9/3
# R2 기준 0.55 이상
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,  # 디폴트 75%
    test_size=0.3,  # 테스트 30%
    random_state=18,  # 재현성 보장
    shuffle=True,  # 섞기 (기본값)
)

# 2. 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=8))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=70, batch_size=42)


# 3. 결과 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)


def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


print('loss: ', loss)
print('R^2: :', r2)
print('RMSE: ', rmse)

# loss:  0.6237552762031555
# R^2: : 0.5247726098537076
# RMSE:  0.7897816216516536
