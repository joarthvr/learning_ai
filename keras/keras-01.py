import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential() # 모델은 시퀀셜
model.add(Dense(1, input_dim=1)) # 머신러닝

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=5500)

#4. 평가 예측
result1 = model.predict(np.array([4]))
result2 = model.predict(np.array([5]))
print("4의 예측값 : ", result1)
print("5의 예측값 : ", result2)


