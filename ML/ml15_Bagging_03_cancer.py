from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# RandomForestClassifier

#! 1. Data ----------------------------------------------------------
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=1003,
    stratify=y,  # 분류라서 클래스 비율 유지
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#! 2. 모델 ----------------------------------------------------------
#! 배깅(Bagging)은 Bootstrap Aggregating 의 줄임말로, 같은 모델 여러 개를 병렬로 학습시켜 평균(또는 다수결)으로 합치는 앙상블 기법
# model = DecisionTreeClassifier()
model = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=150,
    random_state=1003,
    n_jobs=-1,
    max_samples=0.7,
    bootstrap=True,  # 디폴트 # 중복 허용
    # bootstrap=False,  # 중복 허용 x
)

#! 훈련 -------------------------------------------------------------
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('최종 점수: ', result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc: ', acc)

"""
랜덤포레스트
최종 점수:  0.9824561403508771
acc:  0.9824561403508771

디시젼트리
최종 점수:  0.9912280701754386
acc:  0.9912280701754386
"""
