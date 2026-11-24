# ! 수도 라벨링(Pseudo Labeling) 기법 : 모델을 돌려서 나온 결과로 결측치를 찾아 보충한다.
# ! 스태킹 : 모델을 돌려 나온 예측값으로 컬럼을 구성해서 새로운 데이터셋을 만든다.
# !         같은 데이터로 여러 모델을 돌리고, 돌리는 족족 그 예측값이 컬럼으로 하나씩 추가된다.

# ! 스태킹(Stacking)은 서로 다른 모델 여러 개의 '예측값'을 새로운 입력(피처)으로 삼아,
# ! 그 위에 올린 '메타 모델'이 최종 예측을 하도록 학습시키는 앙상블 기법

# ============================================================
# ? 동작 순서
# ============================================================
# ? 1단계 (베이스 모델): XGB, RF, LGBM, CatBoost 처럼 서로 다른 모델이 각자 학습하고 예측한다.
# ? 2단계 (메타 모델)  : 1단계의 예측값들을 컬럼으로 이어 붙인 새 데이터를 입력으로 받아,
# ?                      '어떤 모델을 얼마나 믿을지' 를 학습해서 최종 예측을 낸다.
# ?
# ?   x_train ─┬→ XGB ─→ 예측1 ┐
# ?            ├→ RF  ─→ 예측2 ├→ [예측1, 예측2, 예측3, 예측4] ─→ 메타 모델 ─→ 최종 예측
# ?            ├→ LGBM ─→ 예측3 ┤
# ?            └→ CAT ─→ 예측4 ┘

# ============================================================
# ? 왜 교차검증(K-fold)이 필요한가
# ============================================================
# ? - 베이스 모델이 '학습에 쓴 데이터' 를 다시 예측하면 정답을 이미 봐서 너무 잘 맞힌다.
# ? - 그 예측값으로 메타 모델을 학습하면 베이스 모델을 과대평가하게 된다 (데이터 누수, 과적합).
# ? - 그래서 훈련 데이터를 K 조각으로 나눠, 각 조각을 '그 조각을 안 본 모델' 로 예측한
# ?   값(out-of-fold 예측)을 메타 모델의 학습 데이터로 쓴다.
# ? - sklearn 의 StackingClassifier / StackingRegressor 는 cv 인자(기본 5)로 이 과정을 알아서 한다.

# ============================================================
# ? 보팅과의 차이
# ============================================================
# ? | 구분        | 보팅                       | 스태킹                           |
# ? | 합치는 주체 | 고정된 규칙 (다수결, 평균) | 메타 모델이 합치는 법을 학습     |
# ? | 가중치      | 사람이 weights 로 지정     | 데이터에서 자동으로 학습         |
# ? | 학습 비용   | 모델 수만큼                | 모델 수 x K-fold + 메타 모델     |
# ? | 위험        | 거의 없음                  | 누수, 과적합 (교차검증으로 방지) |
# ?
# ? - 보팅은 '평균', 스태킹은 '평균내는 법까지 학습' 한다고 보면 된다.
# ? - 배깅은 같은 모델 + 데이터를 다르게, 부스팅은 앞 모델의 오차를 보완,
# ?   스태킹은 서로 다른 모델 + 메타 모델 로 합친다.

# ============================================================
# ? sklearn 구현
# ============================================================
# ? - 분류: StackingClassifier(estimators=[...], final_estimator=..., cv=5)
# ? - 회귀: StackingRegressor(estimators=[...], final_estimator=..., cv=5)
# ? - estimators      : ('이름', 모델) 튜플 리스트 (보팅과 같은 형식)
# ? - final_estimator : 메타 모델. 기본값은 분류 LogisticRegression, 회귀 RidgeCV.
# ?                     메타 모델은 단순한 모델(선형)이 잘 맞는다. 입력이 이미 좋은 예측값이라
# ?                     복잡한 모델을 올리면 오히려 과적합하기 쉽다.
# ? - passthrough=True: 예측값뿐 아니라 원본 피처도 메타 모델에 같이 넘긴다 (기본 False).
# ? - 분류에서 stack_method='auto' 면 predict_proba 가 있는 모델은 확률을, 없으면 클래스를 넘긴다.

# ============================================================
# ? 컴피티션(Kaggle, Dacon 등) 에서의 스태킹
# ============================================================
# ? - 컴피티션 상위권에서 마지막 단계로 많이 쓴다. 순위를 가르는 건 소수점 아래 차이라서
# ?   스태킹으로 얻는 작은 이득이 등수에 영향을 준다.
# ? - 성능이 극적으로 오르는 기법은 아니다. 단일 모델 대비 조금씩(짜잘하게) 올라가는 정도다.
# ?   피처 엔지니어링이나 데이터 정제가 먼저고, 스태킹은 그 위에 얹는 마무리 튜닝이다.
# ? - 이득이 작은 만큼 학습 비용(베이스 모델 x K-fold) 대비 효과를 따져서 쓴다.

# ============================================================
# ? 장단점 / 주의
# ============================================================
# ? + 모델마다 잘하는 영역이 달라서, 잘 짜면 단일 모델이나 보팅보다 성능이 오른다.
# ? - 구조가 복잡하고 학습이 오래 걸린다 (베이스 모델 x K-fold).
# ? - 샘플이 적은 데이터(diabetes 442개 등) 에서는 이득이 작거나 오히려 나빠질 수 있다.
# ? - 베이스 모델들이 비슷하게 틀리면 메타 모델이 할 수 있는 게 없다. 종류가 다른 모델을 고른다.
# ? - 스케일에 민감한 모델(KNN, SVM, 로지스틱)이 섞이면 MinMaxScaler 를 먼저 적용한다.


import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier,
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

MODELS = [xgb, lg, rf, cat]
TRAIN_LIST = []
TEST_LIST = []

for model in MODELS:
    #! 3. 훈련 -------------------------------------------------------------
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)

    y_train_pred = model.predict(x_train)  # 메타 모델의 학습용 입력 컬럼
    y_pred = model.predict(x_test)  # 메타 모델의 평가용 입력 컬럼

    TRAIN_LIST.append(y_train_pred)
    TEST_LIST.append(y_pred)

    score = accuracy_score(y_test, y_pred)
    class_name = model.__class__.__name__
    print('{0} ACC: {1:4f}'.format(class_name, score))


# 모델별 예측이 행으로 쌓여 (4, 샘플수) 이므로 .T 로 (샘플수, 4) 로 뒤집는다.
# → 모델 하나가 컬럼 하나가 되는 새 데이터셋
x_train_new = np.array(TRAIN_LIST).T
x_test_new = np.array(TEST_LIST).T
print(x_train_new.shape, x_test_new.shape)

#! 2-2. 모델 ----------------------------------------------
# 메타 모델: 베이스 모델들의 예측값(컬럼 4개)으로 진짜 정답 y 를 맞히도록 학습한다.
model2 = CatBoostClassifier(verbose=0)

#! 3-2. 훈련 -----------------------------------------------
model2.fit(x_train_new, y_train)

#! 4. 평가, 예측 -------------------------------------------
y_pred2 = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred2)
print('스태킹 결과 ACC: {0:4f}'.format(score2))

# [주의] x_train 예측은 베이스 모델이 학습에 쓴 데이터를 다시 맞힌 값이라 너무 잘 맞는다 (누수).
#        그래서 메타 모델이 베이스 모델을 과대평가한다. 제대로 하려면 K-fold 의
#        out-of-fold 예측(cross_val_predict, 또는 StackingClassifier)을 써야 한다.


"""
XGBClassifier ACC: 0.938596
LGBMClassifier ACC: 0.964912
RandomForestClassifier ACC: 0.973684
CatBoostClassifier ACC: 0.973684
"""
