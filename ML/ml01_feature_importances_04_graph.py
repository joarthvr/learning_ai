from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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

model = XGBClassifier(
    random_state=random_state,
)

#! 3. 컴파일 훈련 -------------------------------------------------------------------
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('acc', score)
print(model.feature_importances_)

# ? F 스코어 : 빈도수
print('F 스코어', model.feature_importances_)

# ? Gain 스코어 : 손실함수의 개선 결과 높으면 개선 많이 됌
gain_score = model.get_booster().get_score(importance_type='gain')
print('게인 스코어', gain_score)

#! 4. 그래프 화 -------------------------------------------------------------------

import matplotlib.pyplot as plt
from xgboost import plot_importance

# # ? F-score 그래프
# plot_importance(model, importance_type='weight')
# plt.title('F-score')
# plt.show()


# ? Gain-score 그래프
plot_importance(model, importance_type='gain')
plt.title('Gain-score')
plt.show()
