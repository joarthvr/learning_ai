from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,  # 디폴트 75%
    test_size=0.3,  # 테스트 30%
    random_state=42,  # 재현성 보장
    shuffle=True,  # 섞기 (기본값)
)

# 2. 모델 설정
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.30)

# 4. 결과 예측
loss = model.evaluate(x_test, y_test)

print('loss: ', loss)
print('R^2: ', r2_score(y_test, model.predict(x_test)))
print('RMSE: ', root_mean_squared_error(y_test, model.predict(x_test)))
