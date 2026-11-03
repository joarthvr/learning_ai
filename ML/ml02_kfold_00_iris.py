import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# ! 1. Data -----------------------------------------------------------------------------------
iris = load_iris()
x = iris.data
y = iris.target

n_split = 5
kfold = KFold(
    n_splits=n_split,
    shuffle=True,
    random_state=333,
)

# ! 2. Model ---------------------------------------------------------------------------------
model = DecisionTreeClassifier()

# ! 3. Train ---------------------------------------------------------------------------------
scores = cross_val_score(
    model,
    x,
    y,
    cv=kfold,
)

# ! 4. Evaluate -----------------------------------------------------------------------------
print('acc: ', scores, '평균 acc: ', round(np.mean(scores), 2))
