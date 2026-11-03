import warnings

import numpy as np
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')

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

# StandardScaler 는 음수를 만들어 MultinomialNB 등이 오류 → MinMaxScaler (0~1)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# ! 2. Model ---------------------------------------------------------------------------------
classifiers = all_estimators(type_filter='classifier')

print('all models:', classifiers)
print('sk version:', sk.__version__)
print('모델의 개수:', len(classifiers))

n_split = 5
kfold = KFold(
    n_splits=n_split,
    shuffle=True,
    random_state=333,
)

for name, Model in classifiers:
    try:
        model = Model()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(name, '평균 cv acc:', round(np.mean(scores), 4), 'test acc:', round(acc, 4))
    except Exception as e:
        print(name, '모델 오류:', e)
