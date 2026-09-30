import time

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Sequential

# 1. 데이터
PATH = './_save/keras46/horse/'
x_train = np.load(PATH + 'keras46_horse_x_train.npy')
x_test = np.load(PATH + 'keras46_horse_x_test.npy')
y_train = np.load(PATH + 'keras46_horse_y_train.npy', allow_pickle=True)
y_test = np.load(PATH + 'keras46_horse_y_test.npy', allow_pickle=True)

print(x_train.shape, y_train.shape)  # (100, 200, 200, 3) (100,)
print(x_test.shape, y_test.shape)  # (59, 200, 200, 3) (59,)

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(200, 200, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(patience=50, restore_best_weights=True)
mcp = ModelCheckpoint(PATH, save_best_only=True, verbose=1)
start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=16,
    callbacks=[es],
    validation_split=0.3,
)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
print(f'loss = {loss[0]:.4f}')
print(f'acc = {loss[1]:.4f}')
print(f'acc_score = {accuracy_score(y_test, y_pred):.4f}')

# loss = 0.0402
# acc = 1.0000
# acc_score = 1.0000
