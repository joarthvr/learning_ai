from sklearn.datasets import load_diabetes
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=153,  # 재현성 보장
    shuffle=True,  # 섞기 (기본값)
)

# 2. 모델 설정
model = Sequential()
model.add(Dense(3, input_dim=10))
model.add(Dense(9))
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
    epochs=100,
    batch_size=10,
    verbose=1,
    validation_split=0.2,
    callbacks=[earlyStopping],
)


# 4. 결과 예측
loss = model.evaluate(x_test, y_test)
rmse = root_mean_squared_error(y_test, model.predict(x_test))
print(rmse)
# mpl.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트 (맑은 고딕)
# mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# plt.figure(figsize=(9, 6))  # 그래프 사이즈 지정 figsize = (가로, 세로)
# plt.plot(hist.history['loss'][3:], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'][3:], marker='.', c='blue', label='val_loss')
# plt.legend(loc='upper right')  # 우측 상단에 라벨표시
# plt.title('디아베티스 loss')  # 제목 부여
# plt.xlabel('epochs')  # x축 이름 부여
# plt.ylabel('loss')  # y축 이름 부여
# plt.grid()  # 격자 표시 추가
# plt.show()  # 그래프 출력


"""
51.68754644106802
"""
