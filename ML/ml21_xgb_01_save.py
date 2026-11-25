import os

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

#! 1. Data ----------------------------------------------------------
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=1003,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#! 2. 모델 ----------------------------------------------------------
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
    'objective': 'reg:squarederror',  # 회귀. 이진분류는 'binary:logistic', 다중분류는 'multi:softmax'
    'eval_metric': 'rmse',  # 평가 지표. 분류는 'logloss', 'mlogloss', 'auc' 등
    'random_state': 1003,
    # ? 참고: 조기 종료는 fit 에 eval_set(검증 데이터)을 넘길 때만 쓸 수 있다
    # 'early_stopping_rounds': 50,
}
# ! 파이썬의 * 와 **
# ? 정의할 때(묶기) : *args → 튜플,  **kwargs → 딕셔너리
# ? 호출할 때(풀기) : *리스트/튜플 → 순서대로,  **딕셔너리 → 키=값 인자
# ? 값 두 개 사이에 있으면 산술 : 3 * 4 = 12,  2 ** 3 = 8
model = XGBRegressor(
    # **params,
)  # XGBRegressor(max_depth=6, ...) 와 같다
model.set_params(
    **params,
    early_stopping_rounds=50,
)
#! 3. 훈련 -------------------------------------------------------------
model.fit(
    x_train,
    y_train,
    eval_set=[(x_test, y_test)],
    verbose=1,
)
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2: ', r2)

#! 5. 저장 -------------------------------------------------------------
PATH_SAVE = './_save/xgb/'
os.makedirs(PATH_SAVE, exist_ok=True)  # 폴더가 없으면 dump 가 실패한다
# 학습이 끝난 모델 객체를 통째로 파일로 저장
model.save_model(PATH_SAVE + 'm21_xgb_save.dat')
print('저장 완료: ', PATH_SAVE + 'ml21_california.xgb')

"""
최종 점수:  0.8500929930444414
r2:  0.8500929930444414
"""
