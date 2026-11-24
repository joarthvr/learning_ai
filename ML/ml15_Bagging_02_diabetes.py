from sklearn.datasets import load_diabetes
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#
# RandomForestRegressor

#! 1. Data ----------------------------------------------------------
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=1003,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#! 2. 모델 ----------------------------------------------------------
# model = DecisionTreeRegressor()
model = BaggingRegressor(
    RandomForestRegressor(),
    n_estimators=150,
    random_state=1003,
    n_jobs=-1,
    max_samples=0.5,
    bootstrap=True,  # 디폴트 # 중복 허용
    # bootstrap=False,  # 중복 허용 x
)

#! 훈련 -------------------------------------------------------------
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

"""
최종 점수:  0.27171660574116663
r2:  0.27171660574116663

최종 점수:  0.33597413023918044
r2:  0.33597413023918044

랜덤 포레스트
최종 점수:  0.38098597541401835
r2:  0.38098597541401835
"""
