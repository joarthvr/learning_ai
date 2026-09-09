# https://www.kaggle.com/competitions/bike-sharing-demand/data
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

########################################################
# 1. 데이터
path = './_data/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)  # [10886 rows x 8 columns]
y = train_csv['count']  # Name: count, Length: 10886, dtype: int64, shape = (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,  # 디폴트 75%
    test_size=0.3,  # 테스트 30%
    random_state=77,  # 재현성 보장
    shuffle=True,  # 섞기 (기본값)
)

########################################################
# 2. 모델 설정
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=8))  # 활섬함수 적용 음수 제거
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


#############################################################
# 3. 컴파일 훈련


######################### EraryStopping ######################
earlyStopping = EarlyStopping(
    monitor='val_loss',  # 모니터링 지표
    patience=15,  # 해당 값의 epoch 동안 개선 없으면 중단
    mode='min',  # 'min'(손실), 'max'(정확도), 'auto'
    restore_best_weights=True,
)
##############################################################

model.compile(loss='mse', optimizer='adam')
hist = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=42,
    verbose=1,
    validation_split=0.30,
    callbacks=[earlyStopping],
)

########################################################
# 4. 결과 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

########################################################
loss = model.evaluate(x_test, y_test)
print('R2: ', r2_score(y_test, model.predict(x_test)))
print('RMSE: ', root_mean_squared_error(y_test, model.predict(x_test)))
#######################################################################

mpl.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트 (맑은 고딕)
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

plt.figure(figsize=(9, 6))  # 그래프 사이즈 지정 figsize = (가로, 세로)
plt.plot(hist.history['loss'][3:], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'][3:], marker='.', c='blue', label='val_loss')
plt.legend(loc='upper right')  # 우측 상단에 라벨표시
plt.title('캐글 바이크 loss')  # 제목 부여
plt.xlabel('epochs')  # x축 이름 부여
plt.ylabel('loss')  # y축 이름 부여
plt.grid()  # 격자 표시 추가
plt.show()  # 그래프 출력

# ===== RESULT =====
