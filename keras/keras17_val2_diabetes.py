

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
model.fit(x_train, y_train, epochs=1000, batch_size=10,
        verbose=1,
        validation_split=0.30)


#3. 결과 예측
loss = model.evaluate(x_test, y_test)

print("loss: ", loss)
print("R^2: ", r2_score(y_test, model.predict(x_test)))
print("RMSE: ", root_mean_squared_error(y_test, model.predict(x_test)))