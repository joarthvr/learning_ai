from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#! 1. 데이터 -----------------------------------------------------------------------

x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)

random_state = 12313

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=random_state,
    stratify=y,
)

#! 2. 모델 구성 ---------------------------------------------------------------------
model1 = DecisionTreeClassifier(
    random_state=random_state,
)
model2 = RandomForestClassifier(
    random_state=random_state,
)
model3 = GradientBoostingClassifier(
    random_state=random_state,
)
model4 = XGBClassifier(
    random_state=random_state,
)
models = [model1, model2, model3, model4]

#! 3. 컴파일 훈련 -------------------------------------------------------------------
for model in models:
    model.fit(x_train, y_train)
    print(f'------------------{model.__class__.__name__}------------------')

    #! 4. 평가, 예측 --------------------------------------------------------------------
    score = model.score(x_test, y_test)
    print(f'acc: {score}')
    print(model.feature_importances_)

"""
(150, 4) (150,)
------------------DecisionTreeClassifier------------------
acc: 1.0
[0.         0.0125     0.58159224 0.40590776]
------------------RandomForestClassifier------------------
acc: 1.0
[0.10569164 0.02632235 0.47931957 0.38866645]
------------------GradientBoostingClassifier------------------
acc: 1.0
[0.00190809 0.01628116 0.58551877 0.39629198]
------------------XGBClassifier------------------
acc: 1.0
[0.0107842  0.01729211 0.78464234 0.18728137]
"""
