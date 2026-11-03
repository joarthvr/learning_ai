import warnings

import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# ! 2. Model ---------------------------------------------------------------------------------
all = all_estimators(type_filter='classifier')

print('all models:', all)
print('sk version:', sk.__version__)
print('모델의 개수:', len(all))

for name, model in all:
    try:
        print(name)
        model = model()
        # 훈련
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(name, '의 정답률: ', acc)
    except Exception as e:
        print(name, '모델 오류:', e)
