import numpy as np
from tensorflow.keras.models import load_model

"""
개고양이 가중치를 가져와서 모델 완성
데이터는 개 고양이 npy 데이터 사용
내 사진도 npy 불러와서 predict 진행
"""

PATH = 'C:/study/_data/kaggle_cat_dog_npy/'
data = './_data/kaggle_cat_dog_npy/keras48_cat.npy'
data = np.load(data) / 255
print(data)

# 1. 데이터
x_train = np.load(PATH + 'keras45_01_x_train.npy')
y_train = np.load(PATH + 'keras45_01_y_train.npy')
x_test = np.load(PATH + 'keras45_01_x_test.npy')
y_test = np.load(PATH + 'keras45_01_y_test.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델
model = load_model(f'{PATH}k48_0921_1351-0041-0.4406.keras')

# 3. 예측
y_pred = model.predict(data)
print(y_pred)
