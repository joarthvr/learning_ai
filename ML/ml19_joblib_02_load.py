import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# (짝: ml19_joblib_01_save.py)
PATH_SAVE = './_save/joblib/'

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
# 학습된 상태 그대로 불러오므로 fit 이 필요 없다
model = joblib.load(PATH_SAVE + 'ml19_california.joblib')

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
