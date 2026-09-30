# 48 복사

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

path = 'c:/study/_data/image/'
img = load_img(path + 'chan.jpg', target_size=(150, 150))

print(img)
print(type(img))  # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr.shape)  # (1, 150, 150, 3)
print(type(arr))  # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0)  # 차원 증가 reshape 대신 사용
print(arr.shape)  #

# np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path + 'keras48_me.npy', arr=arr)

# print('Save Completed')

################### 증폭 ###################
datagen = ImageDataGenerator(  # 이 중에 랜덤으로 몇개 적용
    rescale=1.0 / 255,  # . 수차화는 여기, 붙이면 형변환 됐단 표시 (float) = 개발자들의 룰
    horizontal_flip=True,  # 수평 뒤집기 (좌우반전)
    # vertical_flip=True, # 수집 뒤집기 (상하반전)
    width_shift_range=0.1,  # 평행 이동
    # height_shift_range=0.1,
    rotation_range=10,  # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.1,
    # shear_range=0.7, # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode='nearest',
)

it = datagen.flow(
    arr, batch_size=1
)  # flow_from_directory 대신 특정 파일만 쓸떄, flow 한번할때마다 datagen을 한번 적용 (즉, 증폭이 적혀져있는 것 중 랜덤으로 "한 번만" 적용됌)

print(it)  # <keras.preprocessing.image.NumpyArrayIterator object at 0x00000204986CA770>

print(it.next())  # itertor의 1개 요소만 보여줌. 파이썬 3.10까지만 작동 (근데 되긴됌)
print(next(it))  # 위랑 아래랑 같음
print(next(it).shape)  # (1, 150, 150, 3)


fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5, 5))
for i in range(5):
    # batch = it.next()
    batch = next(it)
    # print(batch.shape)
    batch = batch.reshape(150, 150, 3)

    ax[i].imshow(batch)
    # ax[i].axis('off')
plt.show()
