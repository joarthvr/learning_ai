import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import set_random_seed, to_categorical

# =============================================================================
SEED = 42
NUM_WORDS = 1000  # 단어사전의 개수, 빈도수가 높은 단어 순으로 뽑는다 (Embedding input_dim 과 동일)
TEST_SPLIT = 0.2
MAXLEN = 467  # pad_sequences 길이
EMBED_DIM = 100  # Embedding output_dim
LSTM_UNITS = 64
EPOCHS = 100
BATCH_SIZE = 128
VAL_SPLIT = 0.20
PATIENCE = 15
set_random_seed(SEED)

# =============================================================================

#! 1. data -----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=NUM_WORDS,
    # test_split=TEST_SPLIT,
)

print('imdb의 최대 길이: ', max(len(i) for i in x_train))  # ? imdb의 최대 길이:  2494
print('imdb의 최소 길이: ', min(len(i) for i in x_train))  # ? imdb의 최소 길이:  11
print(
    'imdb의 평균 길이: ', sum(map(len, x_train)) / len(x_train)
)  # ?  imdb의 평균 길이:  238.71364

lengths = [len(i) for i in x_train]
print(np.percentile(lengths, [50, 75, 90, 95, 99]))  # ? [178. 291. 467. 610. 926.]

#! 전처리 (패드 시퀀스) -----------------------------------------------------------

# 길이가 제각각인 기사를 (샘플 수, MAXLEN) 직사각형으로 맞춘다. x_test 도 같은 MAXLEN 으로
x_train = pad_sequences(x_train, maxlen=MAXLEN, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, maxlen=MAXLEN, padding='pre', truncating='pre')
print(x_train.shape, x_test.shape)  # ? (25000, 467) (25000, 467)

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
model.add(GRU(LSTM_UNITS))
model.add(Dense(OUTPUT_DIM, activation='softmax'))
model.summary()


#! 3. compile, fit -----------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
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
print('acc  :', acc)  # ? acc 0.6 이상 !!


"""
loss : 0.31734684109687805
acc  : 0.8736799955368042

loss : 0.3210160434246063
acc  : 0.866919994354248

loss : 0.3159169852733612
acc  : 0.8656799793243408
"""
