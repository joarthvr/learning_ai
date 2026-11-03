import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# ! 1. Data -----------------------------------------------------------------------------------
california = fetch_california_housing()
x = california.data
y = california.target

n_split = 5
kfold = KFold(
    n_splits=n_split,
    shuffle=True,
    random_state=42,
)

# ! 2. Model ---------------------------------------------------------------------------------
model = XGBRegressor()

# ! 3. Train ---------------------------------------------------------------------------------
scores = cross_val_score(
    model,
    x,
    y,
    cv=kfold,
)

# ! 4. Evaluate -----------------------------------------------------------------------------
print('r2: ', scores, '평균 r2: ', round(np.mean(scores), 2))
"""
r2:  [0.62992038 0.61235071 0.60165106 0.62396948 0.61498079] 평균 r2:  0.62 // 트리
r2:  [0.80692829 0.8141089  0.80335549 0.82617015 0.80649009] 평균 r2:  0.81 //랜덤포레
r2:  [0.83116009 0.83522882 0.82451226 0.8490521  0.83352332] 평균 r2:  0.83 // XGboost
"""
