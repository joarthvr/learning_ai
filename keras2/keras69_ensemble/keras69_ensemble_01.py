import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

#! 1. data -------------------------------------------------------------
# 삼성 종가   하이닉스 종가
x1_datasets = np.array([range(100), range(301, 401)]).T

# 원유가, 환율, 금 시세
x2_datasets = np.array(
    [
        range(101, 201),
        range(411, 511),
        range(150, 250),
    ]
).T  # (3,100) -> (100,3)

# 화성의 화씨 온도
y = np.array(range(3001, 3101))


# 입력 2개와 y 를 한 번에 분할 (행 짝이 어긋나지 않게), 스케일링보다 먼저
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets,
    x2_datasets,
    y,
    test_size=0.2,
    random_state=42,
)

# 스케일러는 train 에만 fit, test 는 transform 만
scaler1 = StandardScaler()
scaler2 = StandardScaler()
x1_train = scaler1.fit_transform(x1_train)
x2_train = scaler2.fit_transform(x2_train)
x1_test = scaler1.transform(x1_test)
x2_test = scaler2.transform(x2_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

#! 2-1. model1 -------------------------------------------------------------
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(40, activation='relu', name='han4')(dense3)
# model1 = Model(
#     input=input1,
#     outputs=output1,
# )

#! 2-2. model2 -------------------------------------------------------------
input21 = Input(shape=(3,))
dense21 = Dense(50, activation='relu', name='han21')(input21)
dense22 = Dense(40, activation='relu', name='han22')(dense21)
dense23 = Dense(30, activation='relu', name='han23')(dense22)
dense24 = Dense(20, activation='relu', name='han24')(dense23)
output21 = Dense(10, activation='relu', name='han25')(dense24)
# model2 = Model(
#     input=input21,
#     outputs=output21,
# )

#! 2-3. model 합치기 -------------------------------------------------------------
from keras.layers import Concatenate

# concatenate

# 둘다 같은 기능
# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21])

merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)

last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input21], outputs=last_output)
model.summary()

#! 3. compile & fit -------------------------------------------------------------
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8)

#! 4. evaluate, predict -------------------------------------------------------------
# loss 는 스케일된 y 단위
result = model.evaluate([x1_test, x2_test], y_test)
print('loss: ', result)

# test 전체를 예측하고 정답과 비교 (원래 단위로 되돌림)
y_pred = scaler_y.inverse_transform(model.predict([x1_test, x2_test]))
y_true = scaler_y.inverse_transform(y_test)
print('예측값: ', y_pred.flatten())
print('정답: ', y_true.flatten())
