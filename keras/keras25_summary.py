from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# --- 모델 ---
HIDDEN_UNITS = [4, 3]
ACTIVATION = 'relu'
N_CLASSES = 1


# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
for units in HIDDEN_UNITS:
    model.add(Dense(units, activation=ACTIVATION))
model.add(Dense(N_CLASSES))

model.summary()
