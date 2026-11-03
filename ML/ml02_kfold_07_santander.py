import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# ! 1. Data -----------------------------------------------------------------------------------
DATA_PATH = './_data/kaggle_santander/'
train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

n_split = 5
# 이진분류 + 클래스 불균형 → 폴드마다 클래스 비율을 유지하는 StratifiedKFold
kfold = StratifiedKFold(
    n_splits=n_split,
    shuffle=True,
    random_state=42,
)

# ! 2. Model ---------------------------------------------------------------------------------
model = XGBClassifier()

# ! 3. Train ---------------------------------------------------------------------------------
# 불균형이라 accuracy 는 다 0 으로 찍어도 높게 나옴 → roc_auc 로 평가
scores = cross_val_score(
    model,
    x,
    y,
    cv=kfold,
    scoring='roc_auc',
)

# ! 4. Evaluate -----------------------------------------------------------------------------
print('auc: ', scores, '평균 auc: ', round(np.mean(scores), 4))
"""
"""
