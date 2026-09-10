import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer  # 유방암 관련 데이터셋 불러오기
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(datasets.DESCR)  # 데이터셋 설명
print(datasets.feature_names)  # 데이터셋 컬럼명

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)  # (569, 30) (569,)
print(type[x])

print(y)
# 0과 1의 개수가 몇 개인지 찾아보기 - numpy의 unique() 함수 사용
print(np.unique(y))
print(np.unique(y, return_counts=True))  # 0과 1의 개수 확인
# (array([0, 1]), array([212, 357]))

# 0과 1의 개수가 몇 개인지 찾아보기 - pandas의 value_counts() 함수 사용
print(pd.DataFrame(y).value_counts())  # 0과 1의 개수 확인
print(pd.Series(y).value_counts())  # 0과 1의 개수 확인

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,  # 디폴트 75%
    test_size=0.3,  # 테스트 30%
    random_state=77,  # 재현성 보장
    shuffle=True,  # 섞기 (기본값)
    ########################################
    stratify=y,  # y데이터를 기준으로 train과 test를 나눠라 (불균형 데이터셋일 때 사용)
)
print(np.unique(y_train, return_counts=True))  # 0과 1의 개수 확인
print(np.unique(y_test, return_counts=True))  # 0과 1의 개수 확인
# 테스트는 라벨링이 불균형해도 상관없지만, 훈련은 라벨링이 균형잡혀야 한다.

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)

# 2. 모델 구성
model = Sequential()

model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(36, activation='relu'))
# 무조건 이진분류에서는 마지막 레이어의 활성화 함수로 sigmoid를 사용
# 소수점의 가중치를 0~1 사이의 값으로 출력되도록 한다.
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련

# 이진분류에서는 loss로 binary_crossentropy를 사용
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],  # 이진분류에서는 accuracy를 사용 == metrics=['acc']
)

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

hist = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.3,
    verbose=1,
)


# 4. 결과 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

########################################
print('loss: ', round(loss[0], 4))
print('acc: ', round(loss[1], 4))
########################################
y_pred = np.round(y_pred)  # 소수점으로 출력된 값을 반올림하여 0과 1로 변환
print(y_pred[:10])
acc_score = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc_score)

"""
accuracy_score:  0.8947368421052632
"""
