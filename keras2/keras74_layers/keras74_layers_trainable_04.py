from tensorflow.keras.applications import VGG16

model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)
model.trainable = False


model.layers[2].trainable = False
model.summary()

import pandas as pd

pd.set_option('max_colwidth', None)
layers = [(type(layer).__name__, layer.name, layer.trainable) for layer in model.layers]

results = pd.DataFrame(
    layers,
    columns=['layer Type', 'Layer Name', 'Layer Trainable'],
)
print(results)
"""
   layer Type Layer Name  Layer Trainable
0  Functional      vgg16            False
1     Flatten    flatten             True
2       Dense      dense             True
3       Dense    dense_1             True
"""
## 통상적으로 레이어를 동결시켰을 떄 성능이 좋진 못햇다
