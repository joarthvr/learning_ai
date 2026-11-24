# ! 보팅(Voting)은 서로 다른 종류의 모델 여러 개가 각자 예측하고,
# ! 그 결과를 투표로 합치는 앙상블 기법

# ============================================================
# ? 하드 보팅 (voting='hard')
# ? ============================================================
# ? - 각 모델이 낸 '클래스'를 다수결로 합친다.
# ? - 예) A=1, B=1, C=0 → 1이 2표, 0이 1표 → 최종 1
# ? - 확률은 안 보고 라벨만 보기 때문에, 모델이 얼마나 확신했는지는 반영되지 않는다.
# ? - predict_proba 가 없는 모델도 쓸 수 있다.

# ? ============================================================
# ? 소프트 보팅 (voting='soft')
# ? ============================================================
# ? - 각 모델이 낸 '클래스별 확률'을 평균내서, 평균이 가장 큰 클래스를 고른다.
# ? - 예) P(1) = 로지스틱 0.90, KNN 0.60, 트리 0.45 → 평균 0.65 → 최종 1
# ? - 확신의 정도까지 반영해서 보통 하드 보팅보다 성능이 좋다.
# ? - 조건: 모든 모델이 predict_proba 를 지원해야 한다.
# ?   (SVC 는 기본값에서 지원하지 않아서 probability=True 가 필요하다)
# ? - weights=[2, 1, 1] 처럼 믿을 만한 모델에 가중치를 줄 수 있다 (가중 평균).
# ? - 트리 계열은 확률이 0 이나 1 쪽으로 극단적으로 나오기 쉬워서 평균에 미치는 영향이 크다.

# ? ============================================================
# ? 비교
# ? ============================================================
# ? | 구분      | 하드 보팅        | 소프트 보팅                |
# ? | 합치는 값 | 예측한 클래스    | 클래스별 확률              |
# ? | 합치는 법 | 다수결           | 확률 평균 (가중 평균 가능) |
# ? | 조건      | 없음             | predict_proba 필요         |
# ? | 회귀      | 해당 없음        | 해당 없음 (VotingRegressor 는 예측값 평균) |
# ?
# ? - 배깅은 같은 모델 + 데이터를 다르게, 보팅은 다른 모델 + 같은 데이터.
# ? - 서로 다른 종류의 실수를 하는 모델끼리 묶어야 효과가 있다.
# ? - 스케일에 민감한 모델(KNN, SVM, 로지스틱)이 섞이므로 MinMaxScaler 를 먼저 적용한다.


from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
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
#! 배깅(Bagging)은 Bootstrap Aggregating 의 줄임말로, 같은 모델 여러 개를 병렬로 학습시켜 평균(또는 다수결)으로 합치는 앙상블 기법
# model = DecisionTreeClassifier()

xgb = XGBClassifier()
lg = LGBMClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[
        ('XGB', xgb),
        ('RF', rf),
        ('LG', lg),
        ('CAT', cat),
    ],
    # voting='hard',  # default
    voting='soft',
)

#! 훈련 -------------------------------------------------------------
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc: ', acc)

"""
하드
최종 점수:  0.9736842105263158
acc:  0.9736842105263158

소프트
최종 점수:  0.9473684210526315
acc:  0.9473684210526315
"""
