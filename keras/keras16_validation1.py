import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
##################### validation ############################
x_val = np.array([7.8])
y_val = np.array([7.8])
#############################################################
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])



#2. 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(24))
model.add(Dense(36))
model.add(Dense(48))
model.add(Dense(36))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4,
          verbose=1,
          validation_data=(x_val, y_val) # validation_data : 검증데이터를 넣어주면 훈련과정에서 loss와 val_loss가 같이 출력됨
          # 판단 기준
          # loss가 낮아지는데 val_loss가 높아지면 과적합이 발생했다고 판단
          )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(np.array([11])) #추론 예측

print(loss)
print(result)