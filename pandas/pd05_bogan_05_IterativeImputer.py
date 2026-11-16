import numpy as np

# 실험 기능이라 먼저 켜야 IterativeImputer 를 import 할 수 있다
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

import pandas as pd

data = pd.DataFrame(
    [
        [2, np.nan, 6, 8, 10, 8],
        [2, 4, np.nan, 8, np.nan, 4],
        [2, 4, 6, 8, 10, 12],
        [np.nan, 4, np.nan, 8, np.nan, 8],
    ]
).T
data.columns = ['x1', 'x2', 'x3', 'x4']  # .T 로 열이 4개가 됐으므로 이름도 4개

"""
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
5   8.0  4.0  12.0  8.0
"""

xgb = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    random_state=333,
)

imputer = IterativeImputer(
    estimator=xgb,
    max_iter=10,
    tol=0.01,  # ? 디폴트 (얼리스탑의 기준)
    random_state=333,
)

data2 = imputer.fit_transform(data)
print(data2)

"""
[[ 2.          2.          2.          7.99865484]
 [ 9.98037434  4.          4.          4.        ]
 [ 6.          2.01480126  6.          7.99865484]
 [ 8.          8.          8.          8.        ]
 [10.          7.97927809 10.          4.01578808]
 [ 8.          4.         12.          8.        ]]
"""
print('실제 반복 횟수: ', imputer.n_iter_)

# ? 실제 반복 횟수: 3 # 자동으로 얼리스탑핑까지 함
