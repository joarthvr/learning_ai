import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 2. 모델 설정
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

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
    epochs=1000,
    batch_size=4,
    verbose=1,
    validation_split=0.30,
    callbacks=[earlyStopping],
)


# 4. 결과 예측
loss = model.evaluate(x_test, y_test)
print('R2: ', r2_score(y_test, model.predict(x_test)))
print('RMSE: ', root_mean_squared_error(y_test, model.predict(x_test)))


mpl.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트 (맑은 고딕)
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

plt.figure(figsize=(9, 6))  # 그래프 사이즈 지정 figsize = (가로, 세로)
plt.plot(hist.history['loss'][3:], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'][3:], marker='.', c='blue', label='val_loss')
plt.legend(loc='upper right')  # 우측 상단에 라벨표시
plt.title('보스턴 loss')  # 제목 부여
plt.xlabel('epochs')  # x축 이름 부여
plt.ylabel('loss')  # y축 이름 부여
plt.grid()  # 격자 표시 추가
plt.show()  # 그래프 출력

# ===== RESULT  =====
# | 8-7-8-1 | 42 | 100 | 15 | X | 0.2841 | 152.09 | 23891.6 | ES미발동 |
# ===================================
