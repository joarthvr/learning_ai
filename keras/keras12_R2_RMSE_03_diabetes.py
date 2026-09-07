# 9/3
# R2 기준 0.62 이상
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import numpy as np
from sklearn.datasets import load_diabetes
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
    random_state=188,      # 재현성 보장
    shuffle=True          # 섞기 (기본값)
)

#2. 모델 설정
model = Sequential()
model.add(Dense(3, input_dim=10))
model.add(Dense(6))
# model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)


#3. 결과 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("loss: ", loss)
print('R^2: :', r2)
print("RMSE: ", rmse)

# loss:  2183.726318359375
# R^2: : 0.6428795771692073
# RMSE:  46.730357481333776