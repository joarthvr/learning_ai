import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

path = 'c:/study/_data/image/'
img = load_img(path + 'dog.jpg', target_size=(100, 100))

print(img)
print(type(img))  # <class 'PIL.Image.Image'>

plt.imshow(img)
plt.show()

arr = img_to_array(img)
print(arr.shape)  # (200, 200, 3)
print(type(arr))  # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0)  # 차원 증가 reshape 대신 사용
print(arr.shape)

np_path = './_data/kaggle_cat_dog_npy/'

np.save(np_path + 'keras48_dog.npy', arr=arr)

print('Save Completed')
