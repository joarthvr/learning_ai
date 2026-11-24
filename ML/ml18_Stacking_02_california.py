from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

#! 1. Data ----------------------------------------------------------
x, y = fetch_california_housing(return_X_y=True)
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

model = StackingRegressor(
    estimators=[
        ('XGB', xgb),
        ('RF', rf),
        ('LG', lg),
        ('CAT', cat),
    ],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=-1,
    cv=5,  # K-fold 로 out-of-fold 예측을 만들어 메타 모델에 넘긴다 (누수 방지)
)

#! 3-2. 훈련 ----------------------------------------------------------
model.fit(x_train, y_train)

#! 4. 평가, 예측 ------------------------------------------------------
y_pred = model.predict(x_test)
score_new = r2_score(y_test, y_pred)
print('스태킹 결과: ', score_new)
"""
스태킹 결과:  0.8564931391996841
"""
