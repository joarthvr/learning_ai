import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ! 1. Data -----------------------------------------------------------------------------------
iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
    x_train,
    y_train,
    cv=kfold,
)

# ! 4. Evaluate -----------------------------------------------------------------------------
print('acc: ', scores, '평균 acc: ', round(np.mean(scores), 2))

y_predict = cross_val_predict(
    model,
    x_test,
    y_test,
    cv=kfold,
)
print(y_predict, y_test)

acc = accuracy_score(y_test, y_predict)
print(acc)
