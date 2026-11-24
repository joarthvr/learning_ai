# ! 회귀의 보팅: VotingRegressor
# ? - [주의] 회귀(Regressor)는 하드 보팅이 불가능하다.
# ?   하드 보팅은 '클래스'를 다수결로 정하는 방식인데, 회귀에는 클래스가 없고 연속값만 나오기 때문이다.
# ?   그래서 VotingRegressor 에는 voting 인자 자체가 없다 (voting='hard' 를 넘기면 TypeError).
# ?   소프트 보팅도 '확률'이 필요한 방식이라 해당이 없고, 값 평균 한 가지뿐이다.
# ? - 각 모델이 낸 '예측값'을 그대로 평균낸다. (weights 로 가중 평균 가능)
# ? - 분류의 소프트 보팅이 '확률 평균'이라면, 회귀는 '값 평균'이다.

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

#! 1. Data ----------------------------------------------------------
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    random_state=1003,
    # stratify=y,  # 회귀에는 쓰지 않는다 (분류 전용)
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#! 2. 모델 ----------------------------------------------------------
# ? 서로 다른 종류의 모델을 묶어야 효과가 있다 (같이 틀리면 평균을 내도 소용없다).
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
최종 점수:  0.8521821216380988
r2:  0.852182121638098
"""
