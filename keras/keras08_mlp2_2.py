import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
x = np.array(range(1,11))
x = np.array([range(10), range(21,31,), range(201, 211)]).T # (3,10) => (10,3)

y = np.array(range(1,11))


#2. 모델 구성
model = Sequential()
model.add(Dense(12, input_dim=3))
model.add(Dense(16))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=3)

#4. 평가 예측
loss = model.evaluate(x,y)
results = model.predict(np.array([[10,31,211]])) # (1,2)
print(loss)
print(results)

# 6.289486154109625e-10
# [[10.999975]]


# 6.675691111013293e-10
# [[11.000007]]