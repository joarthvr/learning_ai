from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# (짝: ml21_xgb_01_save.py)
PATH_SAVE = './_save/xgb/'

#! 1. Data ----------------------------------------------------------
# 분할과 스케일링은 save 파일과 똑같아야 같은 점수가 나온다
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=1003,
)
# save 파일에서 검증 세트를 뗀 뒤 스케일러를 맞췄으므로 여기서도 똑같이 뗀다
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.8,
    random_state=1003,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#! 2. 모델 불러오기 --------------------------------------------------
# 빈 모델을 만든 뒤 load_model 로 학습 결과를 채운다
model = XGBRegressor()
model.load_model(PATH_SAVE + 'ml21_california.ubj')

#! 4. 평가, 예측 ------------------------------------------------------
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)
"""
최종 점수:  0.8522148839008947
r2:  0.8522148839008947
"""
