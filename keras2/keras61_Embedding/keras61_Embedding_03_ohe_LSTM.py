## ohe 적용

import time
from typing import cast

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import set_random_seed

# ===================================== 상수 =======================================
SEED = 42
MAXLEN = 5  # 가장 긴 문장이 5토큰이라 5 로 맞춤
LSTM_UNITS = [32, 16, 8]
DROPOUT = 0.2
EPOCHS = 500
BATCH_SIZE = 4
VAL_SPLIT = 0.2
set_random_seed(SEED)
STEP = 'Embedding_LSTM'
# ==================================================================================

#! 1. data -----------------------------------------------------------
docs = [
    '너무 재밌다',
    '참 최고에요',
    '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다',
    '한 번 더 보고 싶어요',
    '글쎄',
    '별로에요',
    '생각보다 지루해요',
    '연기가 어색해요',
    '재미없어요',
    '너무 재미없다',
    '참 재밌네요',
    '개똥이 바보',
    '말똥이 잘생겼다',
    '길동이 또 구라친다',
]

label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0])

token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)

padded_x = pad_sequences(x, padding='pre', maxlen=MAXLEN)
print(padded_x.shape)  # ? (15, 5)

x_train, x_test, y_train, y_test = train_test_split(
    padded_x,
    label,
    test_size=0.2,
    random_state=SEED,
    stratify=label,  # 분류이므로 긍정/부정 비율 유지
)
# ===================================== 원핫인코더 =======================================

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))  # test 는 train 으로 배운 인코더로 transform 만

# ==================================================================================

print(y_train.shape, y_test.shape)  # ? (12, 2) (3, 2)

# LSTM 은 (samples, timesteps, features) 3차원 입력이 필요하다.
# 단어 번호 하나를 feature 1개로 보고 (n, 5) → (n, 5, 1) 로 바꾼다.
x_train = x_train.reshape(-1, MAXLEN, 1)
x_test = x_test.reshape(-1, MAXLEN, 1)

#! 2. 모델 구성 -----------------------------------------------------------
model = Sequential()
model.add(Input(shape=(MAXLEN, 1)))
for i, units in enumerate(LSTM_UNITS):
    # 뒤에 LSTM 이 또 오면 시퀀스를 그대로 넘겨야 하므로 마지막 층만 False
    is_last = i == len(LSTM_UNITS) - 1
    model.add(LSTM(units, return_sequences=not is_last))
    model.add(Dropout(DROPOUT))
model.add(Dense(2, activation='softmax'))  # 원핫 라벨 (n, 2) 이므로 클래스 수만큼

#! 3. 컴파일, 훈련 -----------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=VAL_SPLIT,
)
end_time = time.time()

#! 4. 평가, 예측 -----------------------------------------------------------
loss = model.evaluate(x_test, y_test, verbose=0)
y_predict = np.argmax(model.predict(x_test), axis=1)
y_test_label = np.argmax(cast(np.ndarray, y_test), axis=1)
acc = accuracy_score(y_test_label, y_predict)

print('')
print('===== RESULT =====')
print(
    f'| step={STEP} | seed={SEED} | maxlen={MAXLEN} | units={LSTM_UNITS} '
    f'| drop={DROPOUT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'|| acc={acc:.4f} | loss={loss[0]:.4f} '
    f'| time={end_time - start_time:.1f}s |'
)

#! 5. 새 문장 예측 -----------------------------------------------------------
new_docs = ['개똥이 잘생겼다', '글쎄 연기 잘한다', '생각보다 또 구라친다', '한번 더 재미없어요']

new_x = token.texts_to_sequences(new_docs)  # [[24, 27]]
new_x = pad_sequences(new_x, padding='pre', maxlen=MAXLEN)  # [[0 0 0 24 27]]

new_x = new_x.reshape(-1, MAXLEN, 1)  # (4, 5) → (4, 5, 1)
new_pred = model.predict(new_x)

print(new_pred)
for doc, p in zip(new_docs, new_pred):
    # p = [부정 확률, 긍정 확률] (OneHotEncoder 가 0, 1 순으로 열을 만든다)
    print(doc, '→', f'{p[1]:.3f}', '긍정' if np.argmax(p) == 1 else '부정')
"""
===== RESULT =====
| step=Embedding_LSTM | seed=42 | maxlen=5 | units=[32, 16] | drop=0.2 | bs=4 | ep=100
|| acc=0.6667 | loss=0.8036 | time=9.8s |
1/1 [==============================] - 0s 15ms/step
[[0.8531355  0.14686444]
 [0.75637996 0.24362007]
 [0.7521548  0.24784517]
 [0.6176204  0.38237956]]
개똥이 잘생겼다 → 0.147 부정
글쎄 연기 잘한다 → 0.244 부정
생각보다 또 구라친다 → 0.248 부정
한번 더 재미없어요 → 0.382 부정
"""
