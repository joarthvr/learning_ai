# 54-1 복사

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.models import Sequential

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x =  np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ]) # 시간 상 데이터 수기로 분리 [8,9,10] 은 학습시킬 데이터가 없음으로 생략

y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape) # (7, 3) (7,)

x= x.reshape(x.shape[0], x.shape[1], 1) # 시계열 데이터는 3차원 필요 = 
print(x.shape)  # (7, 3, 1)

# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1))) # 행 무시 열 우선
model.add(LSTM(10, input_shape=(3,1)))  # [N, 3, 1] [batch_size, timestamp, feature]
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 5)                 35         # units*units * units*feature * units*bias
#  dense (Dense)               (None, 7)                 42         # (5*5) + (5*1) + (5*1)
#  dense_1 (Dense)             (None, 1)                 8         
# =================================================================
# Total params: 85
# Trainable params: 85
# Non-trainable params: 0
