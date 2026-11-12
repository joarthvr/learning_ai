import time
import warnings

from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
N_SPLITS = 5
#! 1. Data ----------------------------------------------------------------
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8,
)
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=333)

parameters = [
    {
        'n_estimators': [100, 500],
        'max_depth': [6, 10, 12],
        'learning_rate': [0.1, 0.01, 0.001],
    },  # ? 18
    {
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.1, 0.01, 0.001],
    },  # ? 12
    {
        'min_child_weight': [2, 3, 5, 10],
        'learning_rate': [0.1, 0.01, 0.001],
    },  # ? 12
]  # ? 42

#! 2. Model
xgb = XGBRegressor()
model = GridSearchCV(
    xgb,
    parameters,
    cv=kfold,
    verbose=1,
    n_jobs=-1,
)

#! 3. Fit
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print('최적의 매개변수: ', model.best_estimator_)
print('최적의 파라미터: ', model.best_params_)

#! 4. 평가 예측
print('model.r2: ', model.score(x_test, y_test))
print('best_r2: ', model.best_score_)

print('걸린시간:', round(end - start, 2), '초')
"""
Fitting 5 folds for each of 42 candidates, totalling 210 fits
최적의 매개변수:  XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             feature_weights=None, gamma=None, grow_policy=None,
             importance_type=None, interaction_constraints=None,
             learning_rate=0.01, max_bin=None, max_cat_threshold=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
             max_leaves=None, min_child_weight=10, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=None,
             n_jobs=None, num_parallel_tree=None, ...)
최적의 파라미터:  {'learning_rate': 0.01, 'min_child_weight': 10}
model.r2:  0.35203541143453265
best_r2:  0.37628718516426146
걸린시간: 10.45 초
"""
