import numpy as np
from tensorflow.keras.models import load_model

# 1. 데이터

PATH = './_save/keras46/gender/'
data = './_data/kaggle_cat_dog_npy/keras48_me.npy'
data = np.load(data) / 255
print(data)

# 2. 모델
model = load_model(PATH + 'k48_0921_1534-0016-0.2098.keras')

# 3. 예측
y_pred = model.predict(data)
print(y_pred)
