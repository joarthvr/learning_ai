import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# 1단계:
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.5,
    random_state=42,
)

# 2단계
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.125,
    random_state=42,
)


# 2. 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(24))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_data=(x_val, y_val))

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
r2 = r2_score(y_test, model.predict(x_test))
rmse = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))


##################################################

print('loss : ', loss)
print('r2 : ', r2)
print('rmse : ', rmse)
