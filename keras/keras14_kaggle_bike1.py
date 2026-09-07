# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

########################################################
# 1. 데이터
path = '../_data/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
print(train_csv) # [10886 rows x 11 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(test_csv) # [6493 rows x 8 columns]
submission = pd.read_csv(path + "sampleSubmission.csv", index_col = 0)
print(submission) # [6493 rows x 1 columns]

print(train_csv.shape) # (10886,11)
print(test_csv.shape) # (6439,8)
print(submission.shape) # (6439,1)


##################### 결측치 확인 #######################

print(train_csv.info()) # 결측치 없음
print(test_csv.info()) # 결측치 없음

print(train_csv.describe()) # 

##################### 결측치 확인 #######################
print(train_csv.isna().sum())
print(test_csv.isnull().sum()) # 동일한 의미

x = train_csv.drop(['casual', 'registered', 'count'], axis=1) # [10886 rows x 8 columns]
y = train_csv['count'] # Name: count, Length: 10886, dtype: int64, shape = (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,       # 디폴트 75%
    test_size=0.3,        # 테스트 30%
    random_state=42,      # 재현성 보장
    shuffle=True          # 섞기 (기본값)
)

########################################################
# 2. 모델 설정
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=8)) # 활섬함수 적용 음수 제거
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


########################################################
# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=42)

########################################################
# 4. 결과 예측 
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

########################################################
print("loss: ", loss)
print('R^2: :', r2)
print("RMSE: ", rmse)
########################################################
y_submit = model.predict(test_csv)
submission['count'] = y_submit
########################################################

submission.to_csv(path + 'submit/' + 'submit_0904.csv')