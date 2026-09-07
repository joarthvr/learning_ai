# https://dacon.io/competitions/open/235576/overview/agreement

import numpy as np # 수치 계산용
import pandas as pd # 수치화
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, root_mean_squared_error

#1. 데이터
path = "./_data/ddarung/" #상대경로
train_csv = pd.read_csv(path + "train.csv",  index_col = 0) # 데이터 전처리 index_col = 0 필요없는 데이터라서
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
submission = pd.read_csv(path +'submission.csv', index_col=0)
######### 결측치 처리 1. 이상치 제거 ###########
train_csv = train_csv.dropna()

############ train_csv를 x와 y로 분리 ##########
x = train_csv.drop(['count'], axis=1) # 컬럼 삭제
y = train_csv['count'] # 카운트라는 컬럼만 

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,       # 디폴트 75%
    test_size=0.3,        # 테스트 30%
    random_state=153,      # 재현성 보장
    shuffle=True          # 섞기 (기본값)
)

test_csv = test_csv.fillna(test_csv.mean()) # (715, 9)


#2. 모델 설정

model = Sequential()
model.add(Dense(12, input_dim=9))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=200, batch_size=42,
        verbose=1,
        validation_split=0.30)
#######################################################################


#4. 결과 예측
loss = model.evaluate(x_test, y_test)
print('R2: ' , r2_score(y_test, model.predict(x_test)))
print('RMSE: ', root_mean_squared_error(y_test, model.predict(x_test)))
#######################################################################

mpl.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트 (맑은 고딕)
mpl.rcParams['axes.unicode_minus'] = False     # 마이너스 부호 깨짐 방지

plt.figure(figsize=(9,6)) # 그래프 사이즈 지정 figsize = (가로, 세로)
plt.plot(hist.history['loss'][3:], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'][3:], marker='.', c='blue', label='val_loss')
plt.legend(loc='upper right') # 우측 상단에 라벨표시
plt.title('따릉이 loss') # 제목 부여
plt.xlabel('epochs') # x축 이름 부여
plt.ylabel('loss') # y축 이름 부여
plt.grid() # 격자 표시 추가
plt.show() # 그래프 출력
