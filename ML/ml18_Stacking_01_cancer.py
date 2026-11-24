from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

#! 1. Data ----------------------------------------------------------
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=1003,
    stratify=y,  # 분류라서 클래스 비율 유지
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#! 2. 모델 ----------------------------------------------------------

xgb = XGBClassifier()
lg = LGBMClassifier(verbose=-1)  # 경고 메시지 끄기
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)  # 학습 로그 끄기

model = StackingClassifier(
    estimators=[
        ('XGB', xgb),
        ('RF', rf),
        ('LG', lg),
        ('CAT', cat),
    ],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv=5,
)

#! 3-2. 훈련 ----------------------------------------------------------
model.fit(x_train, y_train)

#! 4. 평가, 예측 ------------------------------------------------------
y_pred = model.predict(x_test)
score_new = accuracy_score(y_test, y_pred)
print('스태킹 결과: ', score_new)
"""
스태킹 결과:  0.9298245614035088
"""
