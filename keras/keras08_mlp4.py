import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1],
             [9,8,7,6,5,4,3,2,1,0]]).T

# 11, 0, -1이 나오면 땡큐


#2 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(24))
model.add(Dense(36))
model.add(Dense(48))
model.add(Dense(36))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(3))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=4)

#4. 평가 예측
loss = model.evaluate(x,y)
results = model.predict(np.array([10])) #
print(loss)
print(results)

# 4.3586618689306356e-13
# [[ 1.0999999e+01  1.1771917e-06 -9.9999923e-01]]

# 8.162082121287995e-13
# [[ 1.1000000e+01  1.8998981e-06 -9.9999934e-01]]