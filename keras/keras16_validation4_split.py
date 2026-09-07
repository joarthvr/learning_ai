from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import pandas as pd
import numpy as np

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.75,  
    random_state=111,
)

# 2. 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(24)) 
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
        verbose=1,
        # 훈련할 때 검증할 데이터셋을 나눌 수 있다
        # 성능적인 차이는 없다
        validation_split=0.33
        )

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
r2 = r2_score(y_test, model.predict(x_test))
rmse = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))


##################################################

print("loss : ", loss)
print("r2 : ", r2)
print("rmse : ", rmse)
