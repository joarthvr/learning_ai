import numpy as np

# 1차원 데이터
x1 = np.array([1,2,3]) # 텐서플로우에게 잘 전달하기 위한 형태 (3,)
print(x1.shape) # (3,)

x2 = np.array([[1,2,3]]) # (1,3)
print(x2.shape)

x3 = np.array([[1,2],[3,4]]) # (2,2)
print(x3.shape)

x4 = np.array([[1,2],[3,4],[5,6]])
print(x4.shape)

x5 = np.array([[[1,2], [2,3], [3,4]]])
print(x5.shape) #(1,3,2)

x6 = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
print(x6.shape) #(2,2,2)

x7 = np.array([[[[[1,2,3,4,5], [6,7,8,9,10]]]]])
print(x7.shape) # (1, 1, 1, 2, 5)

x8 = np.array([[[1,2,3]],[[4,5,6]]])
print(x8.shape) #(2,1,3)

x9 = np.array([[[[1]]], [[[2]]]])
print(x9.shape) # (2,1,1,1)
