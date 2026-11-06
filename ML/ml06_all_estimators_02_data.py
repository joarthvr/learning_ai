# 회귀
import time
import warnings

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

N_SPLITS = 5
warnings.filterwarnings('ignore')

# ! 1. Data -----------------------------------------------------------------------------------
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [california, diabetes]

data_name = ['california', 'diabetes']

# ! 2. Model ---------------------------------------------------------------------------------
regressor = all_estimators(type_filter='regressor')

kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=333)

start = time.time()

for index, (x, y) in enumerate(datasets):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        shuffle=True,
        random_state=333,
        train_size=0.8,
    )
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    results = []
    error_count = 0

    for name, Model in regressor:
        try:
            model = Model()

            # ! 3. 훈련 ---------------------------------------------------------------------
            # error_score='raise': 일부 폴드만 실패해도 nan 으로 채우지 않고 예외 → 오류로 집계
            cv_r2 = cross_val_score(model, x_train, y_train, cv=kfold, error_score='raise').mean()

            # ! 4. 평가 ---------------------------------------------------------------------
            # 최종 평가는 test 로
            model.fit(x_train, y_train)
            test_r2 = model.score(x_test, y_test)

            results.append((name, cv_r2, test_r2))

        except Exception:
            error_count += 1
            continue

    # cv 점수 기준으로 최고 모델 선택 (test 로 고르면 test 에 맞춰 고르게 됨)
    best = max(results, key=lambda r: r[1])

    print(
        data_name[index],
        '| 최고 모델:',
        best[0],
        '| cv r2:',
        round(best[1], 4),
        '| test r2:',
        round(best[2], 4),
        '| 성공/오류:',
        len(results),
        '/',
        error_count,
    )


end = time.time()

print('걸린 시간: ', round(end - start, 2), '초')
