# https://dacon.io/competitions/open/235576/overview/agreement

import numpy as np  # 수치 계산용
import pandas as pd  # 수치화
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
# 데이터를 보자마자 이상치와 결측치를 검사한다
# 파일 경로를 명시한다
path = '../_data/ddarung/'  # 상대경로
# path = "c:/study/_data/ddarung" #절대경로

# 불러온 파일을 수치화해야한다
train_csv = pd.read_csv(
    path + 'train.csv', index_col=0
)  # 데이터 전처리 index_col = 0 필요없는 데이터라서
# 인덱스는 데이터가 아니다
# print(train_csv) # [1459 rows x 11 columns] => [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv) # [715 rows x 9 columns]

submission = pd.read_csv(path + 'submission.csv', index_col=0)
# print(submission) # [715 rows x 1 columns]

print(test_csv.shape)
print(submission.shape)
print(train_csv.shape)

print(train_csv.info())
print(test_csv.info())


print(train_csv.info())

######### 결측치 처리 1. 이상치 제거 ###########
train_csv = train_csv.dropna()
print(train_csv)  # [1328 rows x 10 columns]

############ train_csv를 x와 y로 분리 ##########
x = train_csv.drop(['count'], axis=1)  # 컬럼 삭제
print(x)  # [1328 rows x 9 columns]

y = train_csv['count']  # 카운트라는 컬럼만 y로
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.5,  # 디폴트 75%
    test_size=0.5,  # 테스트 30%
    random_state=72,  # 재현성 보장
    shuffle=True,  # 섞기 (기본값)
)
# 2. 모델 설정

model = Sequential()
model.add(Dense(1024, input_dim=9))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(15))
model.add(Dense(128))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=52)

# 4. 결과 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)


def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


print('loss: ', loss)
print('R^2: :', r2)
print('RMSE: ', rmse)

# loss:  2557.42041015625
# R^2: : 0.5876779111389974
# RMSE:  50.57094309442328

# loss:  2557.5322265625
# R^2: : 0.5876598909998346
# RMSE:  50.57204815949276

# loss:  2568.586181640625
# R^2: : 0.5858776873235042
# RMSE:  50.68122079288788

# loss:  2712.032470703125
# R^2: : 0.5724515544682944
# RMSE:  52.077177908751196

# loss:  2703.5302734375
# R^2: : 0.5737919102004447
# RMSE:  51.99548338091939
