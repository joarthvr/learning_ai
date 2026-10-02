import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import reuters
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import set_random_seed, to_categorical

# =============================================================================
SEED = 42
NUM_WORDS = 1000  # 단어사전의 개수, 빈도수가 높은 단어 순으로 뽑는다 (Embedding input_dim 과 동일)
TEST_SPLIT = 0.2
MAXLEN = 313  # pad_sequences 길이
EMBED_DIM = 100  # Embedding output_dim
LSTM_UNITS = 32
EPOCHS = 100
BATCH_SIZE = 128
VAL_SPLIT = 0.20
PATIENCE = 15
set_random_seed(SEED)
# =============================================================================

#! 1. data -----------------------------------------------------------

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=NUM_WORDS,
    # maxlen=100,  # 단어 개수의 최대 길이 제한
    test_split=TEST_SPLIT,
)

print(x_train)
print(x_train.shape, y_train.shape)  # ? (8982,) (8982,)
print(x_test.shape, y_test.shape)  # ? (2246,) (2246,)
print(y_train)  # ? [ 3  4  3 ... 25  3 25]
print(np.unique(y_train))
# ? [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(type(x_train))  # ? <class 'numpy.ndarray'>
print(type(x_train[0]))  # ? <class 'list'>
print(len(x_train[0]))  # ? 87
print('뉴스 기사의 최대 길이: ', max(len(i) for i in x_train))  # ? 뉴스 기사의 최대 길이:  2376
print('뉴스 기사의 최소 길이: ', min(len(i) for i in x_train))  # ? 뉴스 기사의 최소 길이:  13
print(
    '뉴스 기사의 평균 길이: ', sum(map(len, x_train)) / len(x_train)
)  # ? 뉴스 기사의 평균 길이:  145.5398574927633

lengths = [len(i) for i in x_train]
print(np.percentile(lengths, [50, 75, 90, 95, 99]))  # ? [ 95. 179. 313. 459. 718.]


#! 전처리 (패드 시퀀스) -----------------------------------------------------------

# 길이가 제각각인 기사를 (샘플 수, MAXLEN) 직사각형으로 맞춘다. x_test 도 같은 MAXLEN 으로!
x_train = pad_sequences(x_train, maxlen=MAXLEN, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, maxlen=MAXLEN, padding='pre', truncating='pre')
print(x_train.shape, x_test.shape)  # ? (8982, 459) (2246, 459)

# y 원핫
OUTPUT_DIM = int(y_train.max()) + 1  # 46 : 데이터에서 끌어낸다
y_train = to_categorical(y_train, num_classes=OUTPUT_DIM)
y_test = to_categorical(y_test, num_classes=OUTPUT_DIM)

#! 2. model -----------------------------------------------------------
model = Sequential()
model.add(
    Embedding(
        NUM_WORDS,  # ? input_dim
        EMBED_DIM,  # ? output_dim
    )
)  # (None, MAXLEN) → (None, MAXLEN, EMBED_DIM)
model.add(LSTM(LSTM_UNITS))  # 시퀀스 전체를 읽고 마지막 상태 하나만 내보낸다 → (None, LSTM_UNITS)
model.add(Dense(OUTPUT_DIM, activation='softmax'))  # 46개 주제 중 하나
model.summary()

#! 3. compile, fit -----------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,  # 가장 좋았던 에포크의 가중치로 되돌린다
)
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=[es],
    verbose=1,
)

#! 4. evaluate -----------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)
print('acc  :', acc)  # ? acc 0.67 이상 !!

"""
loss : 1.2983347177505493
acc  : 0.7168298959732056

loss : 1.3019357919692993
acc  : 0.6932324171066284

loss : 1.3556456565856934
acc  : 0.6660730242729187
"""
