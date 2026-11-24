# ! 회귀의 보팅: VotingRegressor (diabetes)
# ? - [주의] 회귀(Regressor)는 하드 보팅이 불가능하다.
# ?   하드 보팅은 '클래스'의 다수결이라 클래스가 없는 회귀에는 쓸 수 없다.
# ?   VotingRegressor 에는 voting 인자가 없고, 예측값의 평균만 낸다.
# ? - diabetes 는 샘플이 442개뿐이라 R² 가 원래 낮고, random_state 에 따라 많이 흔들린다.

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

#! 1. Data ----------------------------------------------------------
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.9,
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

model = VotingRegressor(
    estimators=[
        ('XGB', xgb),
        ('RF', rf),
        ('LG', lg),
        ('CAT', cat),
    ],
)

#! 훈련 -------------------------------------------------------------
model.fit(x_train, y_train)
result = model.score(x_test, y_test)  # 회귀의 score 는 R²
print('최종 점수: ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)
"""
최종 점수:  0.36005425212617226
r2:  0.36005425212617226
"""
