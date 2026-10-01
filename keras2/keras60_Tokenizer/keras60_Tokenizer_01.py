import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = '나는 지금 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer()  # ? 객체 (인스턴스)
token.fit_on_texts([text])

print(token.word_index)
# ? {'마구': 1, '매우': 2, '나는': 3, '지금': 4, '진짜': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
print(token.word_counts)
# ? OrderedDict([('나는', 1), ('지금', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x)
# ? [[3, 4, 5, 2, 2, 6, 7, 8, 1, 1, 1, 1, 9]]
print(len(x))
# ? 1

# 원핫 인코딩 3가지 만들기

# #! 1.pandas
df = pd.DataFrame({'Target': np.array(x).flatten()})
encoded_df = pd.get_dummies(df, columns=['Target'])

print('encoded:', encoded_df)
#! 2. sklearn
data_reshaped = np.array(x).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(data_reshaped)
print(encoded)
# #! 3. keras
data_flat = np.array(x).flatten()
encoded = to_categorical(data_flat)
_, compressed_labels = np.unique(data_flat, return_inverse=True)
encoded_final = to_categorical(compressed_labels)
print(encoded_final)
