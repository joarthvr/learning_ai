# ! 상관계수(Correlation Coefficient)
# ! 두 변수가 '함께 움직이는 방향과 정도' 를 -1 ~ 1 사이의 숫자 하나로 나타낸 값

# ============================================================
# ? 값 읽는 법
# ============================================================
# ? +1 : 한쪽이 커지면 다른 쪽도 정확히 직선으로 커진다
# ?  0 : 직선 관계가 없다
# ? -1 : 한쪽이 커지면 다른 쪽은 정확히 직선으로 작아진다
# ? - 부호는 '방향', 절댓값은 '세기' 다. -0.9 는 0.5 보다 강한 관계다.
# ? - 대략 |r| 0.7 이상 강함, 0.3 ~ 0.7 중간, 0.3 미만 약함 (분야마다 기준이 다르다)

# ============================================================
# ? 종류
# ============================================================
# ? pearson  : 직선 관계를 본다. pandas df.corr() 의 기본값. 이상치에 민감하다
# ? spearman : 값 대신 '순위' 로 계산. 직선이 아니어도 계속 증가/감소하면 잡아낸다
# ? kendall  : 순서쌍이 같은 방향인지 비교. 샘플이 적을 때 쓴다

# ============================================================
# ? 머신러닝에서 왜 보나
# ============================================================
# ? 1) 피처 vs 타깃 : 상관이 크면 예측에 도움이 될 후보
# ? 2) 피처 vs 피처 : 상관이 아주 크면(|r| 0.9 이상 등) 같은 정보가 중복된 것
# ?    - 이를 다중공선성이라 한다. 선형모델은 가중치가 불안정해진다
# ?    - 트리 계열(RF, XGB, LGBM, CatBoost)은 예측 성능에는 영향이 작지만,
# ?      중요도가 중복 피처끼리 나뉘어서 feature_importances_ 해석이 흐려진다
# ? 3) 컬럼 정리 : 중복 피처를 하나 빼거나 합쳐서(PCA 등) 차원을 줄인다

# ============================================================
# ? feature_importances_ 와의 차이
# ============================================================
# ? 상관계수             : 모델 없이 데이터만 본다. 두 변수씩, 방향(+, -)이 있다
# ? feature_importances_ : 모델이 학습한 뒤 나온다. 비선형, 조합 효과 반영, 방향은 없다
# ? → 상관이 낮은 피처도 모델에는 중요할 수 있으니, 상관계수만 보고 지우지 않는다.
# ?   지우기 전/후 점수를 같은 조건(random_state)으로 비교해서 결정한다

# ============================================================
# ? 주의
# ============================================================
# ? - 상관관계는 인과관계가 아니다 (아이스크림 판매량과 익사 사고는 '더위' 때문에 같이 는다)
# ? - 0 이라고 무관한 것은 아니다. y = x^2 처럼 U자 관계는 pearson 이 0 에 가깝게 나온다
# ? - 이상치 하나가 값을 크게 바꾼다 → 이상치 처리(pd07) 를 먼저 보고, 산점도도 확인한다
# ? - 결측치 처리 방식에 따라 값이 달라진다 (corr() 은 NaN 을 쌍별로 제외하고 계산한다)
# ? - 분류 타깃의 클래스 번호(0, 1, 2)가 순서 없는 라벨이면 타깃과의 상관은 의미가 없다
# ? - 컬럼 선택에 쓸 때는 train 데이터로만 계산한다 (test 를 보면 데이터 누수)

# ============================================================
# ? 사용법 (연습해 볼 것)
# ============================================================
# ? df = pd.DataFrame(x, columns=피처이름)
# ? df['target'] = y
# ? corr = df.corr()                      # method='spearman' 으로 바꿀 수 있다
# ? corr['target'].sort_values()          # 타깃과의 상관만 보기
# ? sns.heatmap(corr, annot=True)         # 히트맵 (seaborn, matplotlib 필요)


import os

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd

params = {
    # ? 1) 일반: 어떤 부스터를 어떻게 돌릴지
    'booster': 'gbtree',  # 기본 gbtree / gblinear(선형) / dart(트리 드롭아웃)
    'tree_method': 'hist',  # 분기점 찾는 방식. hist 는 구간화해서 빠르다
    'device': 'cpu',  # GPU 는 'cuda'
    'n_jobs': -1,  # 모든 CPU 코어 사용
    'verbosity': 1,  # 0: 없음, 1: 경고, 2: 정보, 3: 디버그
    # ? 2) 부스팅: 몇 번, 얼마나 크게 학습할지
    'n_estimators': 500,  # 트리 개수 (부스팅 반복 횟수), 기본 100
    'learning_rate': 0.05,  # 트리 하나의 반영 비율 (eta), 기본 0.3. 낮추면 n_estimators 를 늘린다
    # ? 3) 트리 구조: 트리 하나가 얼마나 복잡할지 (과적합 조절)
    'max_depth': 6,  # 트리 최대 깊이, 기본 6
    'min_child_weight': 1,  # 자식 노드의 최소 가중치 합. 클수록 보수적
    'gamma': 0,  # 분기에 필요한 최소 손실 감소량. 클수록 덜 쪼갠다
    'max_bin': 256,  # hist 의 구간 개수
    # ? 4) 샘플링: 데이터와 피처를 일부만 써서 다양성 확보
    'subsample': 0.8,  # 트리마다 사용할 행 비율
    'colsample_bytree': 0.8,  # 트리마다 사용할 피처 비율
    'colsample_bylevel': 1,  # 깊이마다 사용할 피처 비율
    'colsample_bynode': 1,  # 분기마다 사용할 피처 비율
    # ? 5) 규제: 가중치가 커지는 것을 억제
    'reg_alpha': 0,  # L1 규제 (alpha)
    'reg_lambda': 1,  # L2 규제 (lambda), 기본 1
    # ? 6) 학습 목표와 평가
    'objective': 'multi:softprob',  # 회귀. 이진분류는 'binary:logistic', 다중분류는 'multi:softmax'
    'eval_metric': 'mlogloss',  # 평가 지표. 분류는 'logloss', 'mlogloss', 'auc' 등
    'random_state': 1003,
    # ? 참고: 조기 종료는 fit 에 eval_set(검증 데이터)을 넘길 때만 쓸 수 있다
    # 'early_stopping_rounds': 50,
}

#! 1. 데이터 세팅 ----------------------------------------------------
datasets = load_iris()
x = datasets.data  # (150, 4) 꽃받침 길이/너비, 꽃잎 길이/너비
y = datasets.target  # (150,) 품종 0, 1, 2

df = pd.DataFrame(x, columns=datasets.feature_names)
df['target'] = y  # 상관계수를 한 번에 보려고 타깃도 컬럼으로 붙인다

print(np.shape(x), np.shape(y))
print(df.head())


# test 는 마지막 평가에만 쓴다. 분류라서 stratify 로 품종 비율 유지
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=1003, stratify=y
)
# 조기 종료용 검증 세트는 train 에서 뗀다
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=1003, stratify=y_train
)

#! 2. 모델 -----------------------------------------------------------
# iris 는 품종 분류라서 Classifier. 트리 모델이라 스케일링은 생략
model = XGBClassifier(**params, early_stopping_rounds=50)

#! 3. 훈련 -------------------------------------------------------------
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)

#! 4. 평가 -------------------------------------------------------------
print('acc: ', model.score(x_test, y_test))

#! 5. 저장 -------------------------------------------------------------
PATH_SAVE = './_save/joblib/'
os.makedirs(PATH_SAVE, exist_ok=True)  # 폴더가 없으면 dump 가 실패한다
# 학습이 끝난 모델 객체를 통째로 파일로 저장
joblib.dump(model, PATH_SAVE + 'ml22_iris.joblib')
print('저장 완료: ', PATH_SAVE + 'ml22_iris.joblib')
