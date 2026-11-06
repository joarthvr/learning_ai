import time
import warnings

from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators

N_SPLITS = 5
warnings.filterwarnings('ignore')

# ! 1. Data -----------------------------------------------------------------------------------
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]

data_name = ['iris', 'cancer', 'wine', 'digits']

# ! 2. Model ---------------------------------------------------------------------------------
classifiers = all_estimators(type_filter='classifier')

# 그룹 개념이 없는 데이터라 StratifiedGroupKFold(groups 필수) 대신 StratifiedKFold
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=333)

start = time.time()

for index, (x, y) in enumerate(datasets):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        shuffle=True,
        random_state=333,
        train_size=0.8,
        stratify=y,
    )
    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    results = []
    error_count = 0

    for name, Model in classifiers:
        try:
            model = Model()

            # ! 3. 훈련 ---------------------------------------------------------------------
            cv_acc = cross_val_score(model, x_train, y_train, cv=kfold).mean()

            # ! 4. 평가 ---------------------------------------------------------------------
            # 최종 평가는 test 로
            model.fit(x_train, y_train)
            test_acc = model.score(x_test, y_test)

            results.append((name, cv_acc, test_acc))

        except Exception:
            error_count += 1
            continue

    # cv 점수 기준으로 최고 모델 선택 (test 로 고르면 test 에 맞춰 고르게 됨)
    best = max(results, key=lambda r: r[1])

    print(
        data_name[index],
        '| 최고 모델:',
        best[0],
        '| cv acc:',
        round(best[1], 4),
        '| test acc:',
        round(best[2], 4),
        '| 성공/오류:',
        len(results),
        '/',
        error_count,
    )


end = time.time()

print('걸린 시간: ', round(end - start, 2), '초')
