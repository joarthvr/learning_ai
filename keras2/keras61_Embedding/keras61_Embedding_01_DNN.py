import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
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
HIDDEN_UNITS = [64, 32, 16, 8]
DROPOUT = 0.3
EPOCHS = 500
BATCH_SIZE = 4
VAL_SPLIT = 0.2
set_random_seed(SEED)
STEP = 'Embedding_DNN'
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
print(token.word_index)
# ? {'참': 1, '너무': 2, '재밌다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}
x = token.texts_to_sequences(docs)
print(x)
# ? [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
# ======================================== 패딩 ============================================
padded_x = pad_sequences(
    x,
    padding='pre',  # 뒤는 post
    maxlen=MAXLEN,
    truncating='pre',  # 디폴트 값으로 앞이 짤림
)

print(padded_x)
"""
? [
?  [ 0  0  0  2  3]
?  [ 0  0  0  1  4]
?  [ 0  0  1  5  6]
?  [ 0  0  7  8  9]
?  [10 11 12 13 14]
?  [ 0  0  0  0 15]
?  [ 0  0  0  0 16]
?  [ 0  0  0 17 18]
?  [ 0  0  0 19 20]
?  [ 0  0  0  0 21]
?  [ 0  0  0  2 22]
?  [ 0  0  0  1 23]
?  [ 0  0  0 24 25]
?  [ 0  0  0 26 27]
?  [ 0  0 28 29 30]
? ]
"""
print(padded_x.shape)  # ? (15, 5)


x_train, x_test, y_train, y_test = train_test_split(
    padded_x,
    label,
    test_size=0.2,
    random_state=SEED,
    stratify=label,  # 분류이므로 긍정/부정 비율 유지
)

# DNN 만들기

#! 2. 모델 구성 -----------------------------------------------------------
model = Sequential()
model.add(Input(shape=(MAXLEN,)))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(DROPOUT))
model.add(Dense(1, activation='sigmoid'))

#! 3. 컴파일, 훈련 -----------------------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
y_predict = np.round(model.predict(x_test)).flatten()
acc = accuracy_score(y_test, y_predict)

print('')
print('===== RESULT =====')
print(
    f'| step={STEP} | seed={SEED} | maxlen={MAXLEN} | units={HIDDEN_UNITS} '
    f'| drop={DROPOUT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'|| acc={acc:.4f} | loss={loss[0]:.4f} '
    f'| time={end_time - start_time:.1f}s |'
)
print('===================================')
print('')

#! 5. 새 문장 예측 -----------------------------------------------------------
new_docs = ['개똥이 잘생겼다', '글쎄 연기 잘한다', '생각보다 또 구라친다', '한번 더 재미없어요']

new_x = token.texts_to_sequences(new_docs)  # [[24, 27]]
new_x = pad_sequences(new_x, padding='pre', maxlen=MAXLEN)  # [[0 0 0 24 27]]

new_pred = model.predict(new_x)
print('결과:', new_pred[0][0], '긍정' if new_pred[0][0] > 0.5 else '부정')
print('결과:', new_pred[1][0], '긍정' if new_pred[1][0] > 0.5 else '부정')
print('결과:', new_pred[2][0], '긍정' if new_pred[2][0] > 0.5 else '부정')
print('결과:', new_pred[3][0], '긍정' if new_pred[3][0] > 0.5 else '부정')


"""
===== RESULT =====
| step=Embedding_DNN | seed=42 | maxlen=5 | units=[64, 32, 16, 8] | drop=0.2 | bs=32 | ep=1000
|| acc=0.3333 | loss=43.2849 | time=38.6s |
===================================
1/1 [==============================] - 0s 16ms/step
긍정 확률: 1.0
결과: 긍정

===== RESULT =====
| step=Embedding_DNN | seed=42 | maxlen=5 | units=[32, 16] | drop=0.2 | bs=32 | ep=100
|| acc=0.3333 | loss=3.1747 | time=4.9s |
===================================
"""
