import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# [검색] 랜덤으로 테스트 데이터와 검증 데이터를 분류해보자
# train과 test 데이터 섞어서 7:3 나눈다
# 섞어서 7:3 나누기

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,  # 디폴트 75%
    test_size=0.3,  # 테스트 30%
    random_state=42,  # 재현성 보장
    shuffle=True,  # 섞기 (기본값)
)

print(x_train, x_test, y_train, y_test)

# 2. 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(24))
model.add(Dense(36))
model.add(Dense(24))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

# 4. 검증 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(np.array([11]))
print(loss, result)
