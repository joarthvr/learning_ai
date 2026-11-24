import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

import pandas as pd

aaa = np.array(
    [
        [-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
        [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350],
    ]
).T

df = pd.DataFrame(aaa, columns=['x1', 'x2'])


def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])

    print('1사분위: ', quartile_1)
    print('q2: ', q2)
    print('3사분위: ', quartile_3)

    iqr = quartile_3 - quartile_1

    print('IQR', iqr)

    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    return (np.where((data > upper_bound) | (data < lower_bound)), iqr, lower_bound, upper_bound)


outlier_loc1, iqr1, low1, up1 = outlier(df['x1'])
outlier_loc2, iqr2, low2, up2 = outlier(df['x2'])

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
for ax, col, low, up in zip(axes, df.columns, [low1, low2], [up1, up2]):
    ax.boxplot(df[col])
    ax.axhline(up, color='pink', label='upper bound')
    ax.axhline(low, color='pink', label='lower bound')
    ax.set_title(col)
    ax.legend()
plt.show()

#! 이상치를 처리해보자

# ? 1. 이상치를 NaN으로 바꾼다
df = df.astype(float)  # int 열에는 NaN 을 넣을 수 없다 (ValueError)

# np.where 의 결과는 '위치'다. loc 은 '라벨'로 고르므로 df.index[위치] 로 라벨을 바꿔서 쓴다
df.loc[df.index[outlier_loc1[0]], 'x1'] = np.nan  # x1 의 이상치 행
df.loc[df.index[outlier_loc2[0]], 'x2'] = np.nan  # x2 의 이상치 행

# ? 2. NaN 을 채운다

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
aaa2 = imputer.fit_transform(df)
print(aaa2)
"""
[[   3.00728369  100.        ]
 [   2.          200.        ]
 [   3.          -30.        ]
 [   4.          400.        ]
 [   5.          500.        ]
 [   6.          600.        ]
 [   7.          599.92285156]
 [   8.          800.        ]
 [   9.          900.        ]
 [  10.         1000.        ]
 [  11.          210.        ]
 [  12.          420.        ]
 [  10.97631931  350.        ]]
"""
