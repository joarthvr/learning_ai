import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ! 1. Data -----------------------------------------------------------------------------------
california = fetch_california_housing()
x = california.data
y = california.target


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
    x_train,
    y_train,
    cv=kfold,
)

# ! 4. Evaluate -----------------------------------------------------------------------------
print('r2: ', scores, '평균 r2: ', round(np.mean(scores), 2))


y_predict = cross_val_predict(
    model,
    x_test,
    y_test,
    cv=kfold,
)

r2 = r2_score(y_test, y_predict)
print(r2)
"""
r2:  [0.82051867 0.84949948 0.83735824 0.83647492 0.82454425] 평균 r2:  0.83
0.7770005829598877
"""
