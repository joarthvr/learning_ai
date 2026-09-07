from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


#2. 모델 생성
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))



#3. 컴파일 훈련

# 1,2,3 == 1,2,3이다는 한번의 훈련
# 1 = 1, 2 = 2, 3 = 3은 세번의 훈련
# 훈련을 시킬 때
# 훈련을 많이 할 수록 성능이 좋아진다
# 큰 데이터를 받는 다고 가정하면 자르는 과정이 필요 => 성능이 좋아짐, 메모리 사용량을 줄일 수 있음
# 데이터가 크면 클수록 자르는 것이 좋다
# 자르는 작업이 배치

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs= 500, batch_size =1)

#4. 평가, 예측.
loss = model.evaluate(x,y)
print('loss: ', loss)
# result1 = model.predict(np.array([1,2,3,4,5,6,7]))
# print(result1)

