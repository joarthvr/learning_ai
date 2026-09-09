import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = './_data/kaggle_santander/'


# 1. 데이터
train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)
test_csv = pd.read_csv(DATA_PATH + 'test.csv', index_col=0)
submission = pd.read_csv(DATA_PATH + 'sample_submission.csv', index_col=0)

print(train_csv.shape) # (200000, 201)
print(test_csv.shape) # (200000, 200)
print(submission.shape) # (200000, 1)

print(train_csv.isna().sum()) # 결측치 검사 
print(test_csv.isnull().sum()) # 결측치 검사

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

print(np.unique(y, return_counts=True)) 
# (array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,    
    test_size=0.3,        
    random_state=153,      
    shuffle=True,          
########################################
    stratify=y,     # y데이터를 기준으로 train과 test를 나눠라 (불균형 데이터셋일 때 사용)
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=153,
    shuffle=True,
    stratify=y_train,      # ← y가 아니라 y_train 기준
)

# 2. 모델 설정
model = Sequential()
model.add(Dense(256, input_dim=200, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(24, activation='relu'))

# 무조건 이진분류에서는 마지막 레이어의 활성화 함수로 sigmoid를 사용
# 소수점의 가중치를 0~1 사이의 값으로 출력되도록 한다.
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련

# 이진분류에서는 loss로 binary_crossentropy를 사용
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'], # 이진분류에서는 accuracy를 사용 == metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    patience=30,
    mode='min',
    restore_best_weights=True
    )

hist = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=1024,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=[es],
    )

# 4. 결과 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 소수점으로 출력된 값을 반올림하여 0과 1로 변환
acc_score = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc_score)
print(np.unique(y_pred, return_counts=True))

from sklearn.metrics import roc_auc_score, classification_report

y_prob = model.predict(x_test)              # 반올림 전 확률 그대로!
print('AUC:', roc_auc_score(y_test, y_prob))
print(classification_report(y_test, np.round(y_prob)))

#####################################################
y_submit = model.predict(test_csv)
submission['target'] = y_submit.round()
submission.to_csv(DATA_PATH + 'submit/' + 'submit_0908_1.csv')