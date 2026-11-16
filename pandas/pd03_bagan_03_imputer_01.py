import numpy as np

import pandas as pd

data = pd.DataFrame(
    [
        [2, np.nan, 6, 8, 10],
        [2, 4, np.nan, 8, np.nan],
        [2, 4, 6, 8, 10],
        [np.nan, 4, np.nan, 8, np.nan],
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
"""

# KNNImputer, IterativeImputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()  # ? 디폴트 평균값

data2 = imputer.fit_transform(data)
print(data2)
"""
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
"""

imputer2 = SimpleImputer(strategy='mean')
data3 = imputer.fit_transform(data)
print(data3)
"""
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
"""
imputer3 = SimpleImputer(strategy='median')
data4 = imputer.fit_transform(data)
print(data4)
"""
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
"""
