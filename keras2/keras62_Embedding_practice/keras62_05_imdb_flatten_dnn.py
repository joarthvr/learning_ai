import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import set_random_seed

# =============================================================================
SEED = 42
NUM_WORDS = 10000  # 단어사전의 개수 (Embedding input_dim 과 동일). 이전: 1000
MAXLEN = 300  # pad_sequences 길이. 이전: 467
EMBED_DIM = 64  # Embedding output_dim. 이전: 100
HIDDEN_UNITS = [128, 64]  # Dense 은닉층 유닛 수
DROPOUT = 0.3  # 층과 층 사이 출력을 끊는 비율
EPOCHS = 100
BATCH_SIZE = 64  # 이전: 128
VAL_SPLIT = 0.20
PATIENCE = 5  # EarlyStopping. 이전: 15
LR_FACTOR = 0.5  # val_loss 가 멈추면 학습률에 곱할 값
LR_PATIENCE = 2  # 학습률을 낮추기까지 기다리는 에포크 (PATIENCE 보다 작아야 한다)
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
print(x_train.shape, x_test.shape)  # ? (25000, 300) (25000, 300)

# 긍정(1)/부정(0) 이진 분류라 y 를 원핫하지 않고 0/1 그대로 쓴다.

#! 2. model -----------------------------------------------------------
model = Sequential()
model.add(Embedding(NUM_WORDS, EMBED_DIM))  # (None, MAXLEN) → (None, MAXLEN, EMBED_DIM)
# Flatten 대신 단어 벡터를 평균낸다 → (None, EMBED_DIM). 길이에 무관해서 input_length 도 필요 없다
model.add(GlobalAveragePooling1D())
model.add(Dropout(DROPOUT))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(DROPOUT))
model.add(Dense(1, activation='sigmoid'))  # 긍정일 확률 하나
model.summary()

#! 3. compile, fit -----------------------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
)
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=[es, rlr],
    verbose=1,
)

#! 4. evaluate -----------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)
print('acc  :', acc)
print(
    f'| seed={SEED} | words={NUM_WORDS} | maxlen={MAXLEN} | emb={EMBED_DIM} '
    f'| hidden={HIDDEN_UNITS} | drop={DROPOUT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| pat={PATIENCE} | lr_factor={LR_FACTOR} | lr_pat={LR_PATIENCE} '
    f'|| acc={acc:.4f} | loss={loss:.4f} |'
)

"""
| seed=42 | words=10000 | maxlen=300 | emb=64 | hidden=[128, 64] | drop=0.3 | bs=64| ep=100 | pat=5 | lr_factor=0.5 | lr_pat=2
|| acc=0.8824 | loss=0.2843 |

| seed=42 | words=10000 | maxlen=300 | emb=64 | hidden=[128, 64] | bs=64 | ep=100 |pat=5 | lr_factor=0.5 | lr_pat=2
|| acc=0.8804 | loss=0.2897 |

| seed=42 | words=1000 | maxlen=467 | emb=100 | hidden=[128, 64] | bs=128 | ep=100
|| acc=0.8540 | loss=0.3383 |
"""
