import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.9,  # 디폴트 75%
    test_size=0.1,  # 테스트 30%
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
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.30)


# 3. 결과 예측
loss = model.evaluate(x_test, y_test)
print('########################## history ############################')
print(hist)
print('########################## loss ############################')
print(hist.history['loss'])
print('########################## val_loss ############################')
print(hist.history['val_loss'])
#######################################################################

mpl.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트 (맑은 고딕)
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

plt.figure(figsize=(9, 6))  # 그래프 사이즈 지정 figsize = (가로, 세로)
plt.plot(hist.history['loss'][3:], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'][3:], marker='.', c='blue', label='val_loss')
plt.legend(loc='upper right')  # 우측 상단에 라벨표시
plt.title('캘리포니아 loss')  # 제목 부여
plt.xlabel('epochs')  # x축 이름 부여
plt.ylabel('loss')  # y축 이름 부여
plt.grid()  # 격자 표시 추가
plt.show()  # 그래프 출력
