import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ! 1. Data -----------------------------------------------------------------------------------
DATA_PATH = './_data/kaggle_santander/'
train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8,
    stratify=y,  # 클래스 비율 유지 (분류라서 사용 가능)
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
    x_train,
    y_train,
    cv=kfold,
    scoring='roc_auc',
)

# ! 4. Evaluate -----------------------------------------------------------------------------
print('auc: ', scores, '평균 auc: ', round(np.mean(scores), 4))

# auc 는 확률로 계산하므로 predict_proba 의 1 번 열(양성 확률)을 받는다
y_predict = cross_val_predict(
    model,
    x_test,
    y_test,
    cv=kfold,
    method='predict_proba',
)[:, 1]

auc = roc_auc_score(y_test, y_predict)
print(auc)
"""
"""
