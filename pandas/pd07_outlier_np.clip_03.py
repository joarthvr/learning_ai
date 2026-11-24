import matplotlib.pyplot as plt
import numpy as np

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

# ? np.clip: 이상치를 NaN 으로 만들지 않고, 경계값(low~up)으로 잘라서 바꾼다
# 경계보다 작으면 low 로, 크면 up 로 바뀌고 범위 안의 값은 그대로다
df['x1'] = np.clip(df['x1'], low1, up1)
df['x2'] = np.clip(df['x2'], low2, up2)
print(df)

"""
    x1    x2
0   -5   100
1    2   200
2    3   -30
3    4   400
4    5   500
5    6   600
6    7  -400
7    8   800
8    9   900
9   10  1000
10  11   210
11  12   420
12  19   350
"""
