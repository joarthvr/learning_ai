import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (150, 4) (150,)
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50]))

############################ 원핫 인코딩 1 numpy ############################
# 0 -> [1, 0, 0]  /  1 -> [0, 1, 0]  /  2 -> [0, 0, 1]
# softmax(출력 3개) + categorical_crossentropy 는 정답도 (n, 3) 이어야 함
y = to_categorical(y)
print(y.shape)   # (150, 3)
print(y)     # 변환 결과 눈으로 확인/
############################################################################

############################ 원핫 인코딩 2 pandas ############################
# 0 -> [1, 0, 0]  /  1 -> [0, 1, 0]  /  2 -> [0, 0, 1]
# df = pd.DataFrame(datasets.target, columns=['label'])
y = pd.get_dummies(datasets.target, dtype='float32')
print(y)

# print(y.shape)   # (150, 3)
############################################################################

############################ 원핫 인코딩 3 sklearn ############################
# 0 -> [1, 0, 0]  /  1 -> [0, 1, 0]  /  2 -> [0, 0, 1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(datasets.target.reshape(-1, 1))
# print(y.shape)   # (150, 3)
print(y)
##############################################################################

exit()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    random_state=42,
    shuffle=True,
    stratify=y,
)

# print(x_train.shape, x_test.shape) # (105, 4) (45, 4)
# print(y_train.shape, y_test.shape) # (105, 3) (45, 3)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
#################################################
model.add(Dense(3, activation='softmax'))
#################################################

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=8,
    validation_split=0.2,   # val_loss가 있어야 EarlyStopping이 작동함
    callbacks=[es],
)
end_time=time.time()

# 4. 평가 예측
result = model.evaluate(x_test, y_test)
print('loss:', result[0])
print('acc:', round(result[1], 2))

#################################################
y_predict = np.argmax(model.predict(x_test), axis=-1)  # 확률 (45,3) -> 라벨 (45,)
y_test_label = np.argmax(y_test, axis=-1)                 # 원핫 정답 -> 라벨
#################################################

acc = accuracy_score(y_test_label, y_predict)
print('accuracy_score:', acc)
print('걸린 시간:', round(end_time - start_time,2))

# 모델은 학습을 위해 확률이라는 부드러운 언어로 말하도록 만들어졌습니다. argmax는 미분이 안 되므로 모델 바깥, 학습이 다 끝난 뒤에만 쓸 수 있음
# to_categorical은 훈련을 위해 정수를 원핫으로 바꾸고,
# argmax는 채점을 위해 원핫과 확률을 다시 정수로 내립니다.

# 둘 다 정상 흐름이고, 중간에 깨지는 일은 없습니다.

# 원핫으로 바꾸는 것: 정답 y를. 모델 출력이 3칸이라 정답도 3칸이어야 loss를 계산할 수 있어서.

# 되돌리는 것1: 정답 y_test를. sklearn이 정수만 읽어서. (형식 문제)

# 되돌리는 것2: 예측 확률을. 확률은 답이 아니라 골라야 해서. (판단 문제)

# 되돌리지 않는 것: y_train과 evaluate용 y_test. 받는 쪽(keras)이 원핫을 이해해서.