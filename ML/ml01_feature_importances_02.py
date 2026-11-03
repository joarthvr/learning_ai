from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#! 1. 데이터 -----------------------------------------------------------------------

x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)

random_state = 12313

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=random_state,
)

#! 2. 모델 구성 ---------------------------------------------------------------------
model1 = DecisionTreeRegressor(
    random_state=random_state,
)
model2 = RandomForestRegressor(
    random_state=random_state,
)
model3 = GradientBoostingRegressor(
    random_state=random_state,
)
model4 = XGBRegressor(
    random_state=random_state,
)
models = [model1, model2, model3, model4]

#! 3. 컴파일 훈련 -------------------------------------------------------------------
for model in models:
    model.fit(x_train, y_train)
    print(f'------------------{model.__class__.__name__}------------------')

    #! 4. 평가, 예측 --------------------------------------------------------------------
    score = model.score(x_test, y_test)
    print(f'r2: {score}')
    print(model.feature_importances_)

"""
(442, 10) (442,)
------------------DecisionTreeRegressor------------------
r2: -0.04310680113361309
[0.06665966 0.01730147 0.24240713 0.06572168 0.04598395 0.07515203
 0.03275987 0.01569529 0.37777162 0.06054731]
------------------RandomForestRegressor------------------
r2: 0.5119758932196357
[0.05673446 0.0158191  0.28651866 0.0880751  0.04270675 0.05373114
 0.05578192 0.02544974 0.30537312 0.06981002]
------------------GradientBoostingRegressor------------------
r2: 0.49283893902818776
[0.04574826 0.01915334 0.27687617 0.10770591 0.03184389 0.05243981
 0.03641781 0.02170681 0.36259194 0.04551605]
------------------XGBRegressor------------------
r2: 0.297304382598786
[0.03447448 0.06620235 0.15667813 0.06391279 0.03511411 0.06001616
 0.06705785 0.08181974 0.35794765 0.07677672]
"""
