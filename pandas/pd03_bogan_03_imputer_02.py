import numpy as np
from sklearn.impute import SimpleImputer

# KNNImputer, IterativeImputer
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

imputer = SimpleImputer(strategy='most_frequent')
data2 = imputer.fit_transform(data)
print(data2)
"""
[[ 2.  2.  2.  8.]
 [ 8.  4.  4.  4.]
 [ 6.  4.  6.  8.]
 [ 8.  8.  8.  8.]
 [10.  4. 10.  8.]
 [ 8.  4. 12.  8.]]
"""

imputer2 = SimpleImputer(strategy='constant', fill_value=777)
data2 = imputer.fit_transform(data)
print(data2)
"""
[[ 2.  2.  2.  8.]
 [ 8.  4.  4.  4.]
 [ 6.  4.  6.  8.]
 [ 8.  8.  8.  8.]
 [10.  4. 10.  8.]
 [ 8.  4. 12.  8.]]
"""

imputer3 = SimpleImputer(strategy=lambda x: np.max(x))  # pyright: ignore[reportArgumentType]
data4 = imputer3.fit_transform(data)
print(data4)
