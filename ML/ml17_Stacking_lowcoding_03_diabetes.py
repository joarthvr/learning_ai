import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

#! 1. Data ----------------------------------------------------------
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=1003,
    # stratify=y,  # 회귀에는 쓰지 않는다 (분류 전용)
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#! 2. 모델 ----------------------------------------------------------

xgb = XGBRegressor()
lg = LGBMRegressor(verbose=-1)  # 경고 메시지 끄기
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)  # 학습 로그 끄기

MODELS = [xgb, lg, rf, cat]
TRAIN_LIST = []
TEST_LIST = []

for model in MODELS:
    #! 3. 훈련 -------------------------------------------------------------
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)  # 회귀의 score 는 R²

    y_train_pred = model.predict(x_train)  # 메타 모델의 학습용 입력 컬럼
    y_pred = model.predict(x_test)  # 메타 모델의 평가용 입력 컬럼

    TRAIN_LIST.append(y_train_pred)
    TEST_LIST.append(y_pred)

    score = r2_score(y_test, y_pred)
    class_name = model.__class__.__name__
    print('{0} R2: {1:4f}'.format(class_name, score))


# 모델별 예측이 행으로 쌓여 (4, 샘플수) 이므로 .T 로 (샘플수, 4) 로 뒤집는다.
# → 모델 하나가 컬럼 하나가 되는 새 데이터셋
x_train_new = np.array(TRAIN_LIST).T
x_test_new = np.array(TEST_LIST).T
print(x_train_new.shape, x_test_new.shape)

#! 2-2. 모델 ----------------------------------------------
# 메타 모델: 베이스 모델들의 예측값(컬럼 4개)으로 진짜 정답 y 를 맞히도록 학습한다.
model2 = CatBoostRegressor(verbose=0)

#! 3-2. 훈련 -----------------------------------------------
model2.fit(x_train_new, y_train)

#! 4. 평가, 예측 -------------------------------------------
y_pred2 = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred2)
print('스태킹 결과 R2: {0:4f}'.format(score2))

# [주의] x_train 예측은 베이스 모델이 학습에 쓴 데이터를 다시 맞힌 값이라 너무 잘 맞는다 (누수).
#        그래서 메타 모델이 베이스 모델을 과대평가한다. 제대로 하려면 K-fold 의
#        out-of-fold 예측(cross_val_predict, 또는 StackingRegressor)을 써야 한다.


"""
XGBRegressor R2: 0.174126
LGBMRegressor R2: 0.299357
RandomForestRegressor R2: 0.345226
CatBoostRegressor R2: 0.280688
(353, 4) (89, 4)
스태킹 결과 R2: 0.213580
"""
