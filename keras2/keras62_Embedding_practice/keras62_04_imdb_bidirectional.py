import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import set_random_seed

# =============================================================================
SEED = 42
NUM_WORDS = 1000  # 단어사전의 개수 (Embedding input_dim 과 동일)
MAXLEN = 467  # pad_sequences 길이 (리뷰 길이 90백분위)
EMBED_DIM = 100  # Embedding output_dim
GRU_UNITS = 64
EPOCHS = 100
BATCH_SIZE = 128
VAL_SPLIT = 0.20
PATIENCE = 15
set_random_seed(SEED)
# =============================================================================

#! 1. data -----------------------------------------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

lengths = [len(i) for i in x_train]
print(np.percentile(lengths, [50, 75, 90, 95, 99]))  # ? [178. 291. 467. 610. 926.]

#! 전처리 (패드 시퀀스) -----------------------------------------------------------

# 리뷰 길이를 (샘플 수, MAXLEN) 으로 맞춘다. x_test 도 같은 MAXLEN 으로!
x_train = pad_sequences(x_train, maxlen=MAXLEN, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, maxlen=MAXLEN, padding='pre', truncating='pre')
print(x_train.shape, x_test.shape)  # ? (25000, 467) (25000, 467)

# 긍정(1)/부정(0) 이진 분류라 y 를 원핫하지 않고 0/1 그대로 쓴다.

#! 2. model -----------------------------------------------------------
model = Sequential()
model.add(Embedding(NUM_WORDS, EMBED_DIM))  # (None, MAXLEN) → (None, MAXLEN, EMBED_DIM)
model.add(
    Bidirectional(GRU(GRU_UNITS))
)  # 앞→뒤, 뒤→앞 두 방향으로 읽고 마지막 상태를 이어붙인다 → (None, GRU_UNITS * 2)
model.add(Dense(1, activation='sigmoid'))  # 긍정일 확률 하나
model.summary()

#! 3. compile, fit -----------------------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
print('acc  :', acc)
print(
    f'| seed={SEED} | words={NUM_WORDS} | maxlen={MAXLEN} | emb={EMBED_DIM} '
    f'| units={GRU_UNITS} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'|| acc={acc:.4f} | loss={loss:.4f} |'
)

"""
| seed=42 | words=1000 | maxlen=467 | emb=100 | units=64 | bs=128 | ep=100
|| acc=0.8759 | loss=0.3037 |
"""
