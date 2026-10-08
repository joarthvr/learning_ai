import numpy as np
from keras.layers import Concatenate
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

x3_datasets = np.array(
    [
        range(100),
        range(301, 401),
        range(77, 177),
        range(33, 133),
    ]
).T

y = np.array(range(3001, 3101))


(
    x1_train,
    x1_test,
    x2_train,
    x2_test,
    x3_train,
    x3_test,
    y_train,
    y_test,
) = train_test_split(
    x1_datasets,
    x2_datasets,
    x3_datasets,
    y,
    test_size=0.2,
    random_state=42,
)

scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
x1_train = scaler1.fit_transform(x1_train)
x2_train = scaler2.fit_transform(x2_train)
x3_train = scaler3.fit_transform(x3_train)
x1_test = scaler1.transform(x1_test)
x2_test = scaler2.transform(x2_test)
x3_test = scaler3.transform(x3_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

#! 2-1. model1 -------------------------------------------------------------
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(40, activation='relu', name='han4')(dense3)

#! 2-2. model2 -------------------------------------------------------------
input21 = Input(shape=(3,))
dense21 = Dense(50, activation='relu', name='han21')(input21)
dense22 = Dense(40, activation='relu', name='han22')(dense21)
dense23 = Dense(30, activation='relu', name='han23')(dense22)
dense24 = Dense(20, activation='relu', name='han24')(dense23)
output21 = Dense(10, activation='relu', name='han25')(dense24)


#! 2-2. model3 -------------------------------------------------------------
input31 = Input(shape=(4,))
dense31 = Dense(50, activation='relu', name='han31')(input31)
dense32 = Dense(40, activation='relu', name='han32')(dense31)
dense33 = Dense(30, activation='relu', name='han33')(dense32)
dense34 = Dense(20, activation='relu', name='han34')(dense33)
output31 = Dense(10, activation='relu', name='han35')(dense34)

#! 2-4. model 합치기 -------------------------------------------------------------
merge1 = Concatenate(name='mg1')(
    [
        output1,
        output21,
        output31,
    ]
)
merge2 = Dense(20, name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)
merge4 = Dense(5, name='mg4')(merge3)

last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input21, input31], outputs=last_output)

model.summary()

#! 3. compile & fit -------------------------------------------------------------
model.compile(loss='mse', optimizer='adam')
model.fit(
    [x1_train, x2_train, x3_train],
    y_train,
    epochs=100,
    batch_size=8,
)

#! 4. evaluate, predict -------------------------------------------------------------
result = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss: ', result)

y_pred = scaler_y.inverse_transform(model.predict([x1_test[:2], x2_test[:2], x3_test[:2]]))
y_true = scaler_y.inverse_transform(y_test[:2])
print('예측값: ', y_pred.flatten())
print('정답: ', y_true.flatten())
