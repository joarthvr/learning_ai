import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# ! 1. Data -----------------------------------------------------------------------------------
DATA_PATH = './_data/kaggle_bike/'
train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)

# casual + registered = count 이므로 남겨두면 답을 보고 푸는 셈 (타깃 누수)
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

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
"""
