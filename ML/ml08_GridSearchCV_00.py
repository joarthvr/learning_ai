import time
import warnings

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
N_SPLITS = 5
#! 1. Data ----------------------------------------------------------------
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8,
    stratify=y,
)
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=333)

parameters = [
    {
        'n_estimaters': [100, 500],
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
xgb = XGBClassifier()
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
print('model.score: ', model.score(x_test, y_test))
print('best_score: ', model.best_score_)

print('걸린시간:', round(end - start, 2), '초')
"""
Fitting 5 folds for each of 42 candidates, totalling 210 fits
최적의 매개변수:  XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              feature_weights=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimaters=100,
              n_estimators=None, n_jobs=None, ...)
최적의 파라미터:  {'learning_rate': 0.1, 'max_depth': 12, 'n_estimaters': 100}
model.score:  0.9416666666666667
best_score:  0.958947928765002
걸린시간: 15.95 초
"""
