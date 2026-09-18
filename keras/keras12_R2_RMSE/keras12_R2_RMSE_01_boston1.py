# 9/3
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 2. 모델 설정
model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=3)
print('=============================================')
# 3. 결과 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
# r2는 텐서플로우에서 지원해주지 않는 지표 그렇게 때문에 사이킷런에서 임포트한다
# r2는 회귀모델에서 보조지표로 사용될 수 있다
# r2 텐서플로우에서 지원하지 않아서 직접 구현한다
# r2를 위해선 test값과 결과값이 필요하다(수식 참조)
# r2가 음수값이 나오면 잘못된 것
# r2가 1에 근접할 수록 긍정적
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)


def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


print(loss, r2, mse, rmse, RMSE(y_test, y_pred))

# 27.157167434692383 0.6737636709366499
