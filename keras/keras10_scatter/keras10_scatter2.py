import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
x = np.array(range(1, 21))
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])
print(len(x), len(y))


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,  # 디폴트 75%
    test_size=0.3,  # 테스트 30%
    random_state=65,  # 재현성 보장
    shuffle=True,  # 섞기 (기본값)
)

# 2. 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=20))
model.add(Dense(24))
model.add(Dense(36))
model.add(Dense(24))
model.add(Dense(20))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

print('=======================================')

# 4. 검증 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(np.array([11]))

# 그래프 그리기
plt.scatter(x, y)
plt.plot(x, result, color='red')
plt.show()
