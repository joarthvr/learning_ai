import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 지금 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '개똥이는 선생을 괴롭힌다. 말똥이는 못생겼다. 길동이는 마구 마구 더 못생겼다.'

token = Tokenizer()  # ? 객체 (인스턴스)
token.fit_on_texts([text1, text2])

x = token.texts_to_sequences([text1, text2])
print(token.word_index)
print(x)
# ? [[4, 5, 6, 2, 2, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 3, 15, 1, 1, 16, 3]]
print(len(x))
# ? 2


#! 2. sklearn
x = np.concatenate(x)
data_reshaped = np.array(x).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(data_reshaped)
print(encoded.shape)  # ? (23, 16)
