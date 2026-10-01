import numpy as np
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
STEP = '중요'
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

#! 2. model -----------------------------------------------------------
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

model = Sequential()
#! ------------------------ 임베딩 1 -----------------------------------

model.add(
    Embedding(
        input_dim=30,  # ? 단어 사전의 개수
        output_dim=100,  # ?차원
        input_length=5,
    )
)
model.add(SimpleRNN(10))
model.add(Dense(1))
# ? embedding (Embedding)       (None, 5, 100)             3000

# ? simple_rnn (SimpleRNN)      (None, 10)                1110

model.summary()
# ? ID 2: 사과 → [0.12, -0.31, 0.08, ..., 0.27]
# ?               └────── 숫자 100개 ──────┘

#! ------------------------ 임베딩 2 -----------------------------------

model.add(
    Embedding(
        30,  # ? input_dim
        100,  # ? output_dim
        # input_length 명시 안해도 알아서 설정함
    )
)
model.add(SimpleRNN(10))
model.add(Dense(1))
