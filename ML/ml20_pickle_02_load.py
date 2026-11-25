import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# (짝: ml20_pickle_01_save.py)
PATH_SAVE = './_save/pickle/'

#! 1. Data ----------------------------------------------------------
# 분할과 스케일링은 save 파일과 똑같아야 같은 점수가 나온다
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=1003,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#! 2. 모델 불러오기 --------------------------------------------------
# 'rb' = 바이너리 읽기. with 를 쓰면 파일이 자동으로 닫힌다
# [주의] pickle 파일은 여는 순간 코드가 실행될 수 있으니 믿을 수 있는 파일만 불러온다
with open(PATH_SAVE + 'ml20_california.pickle', 'rb') as f:
    model = pickle.load(f)

#! 4. 평가, 예측 ------------------------------------------------------
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)
"""
최종 점수:  0.8500929930444414
r2:  0.8500929930444414
"""
