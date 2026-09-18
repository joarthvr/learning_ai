import numpy as np

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

# [칮아보기] 넘파이 리스트의 슬라이싱 => 7:3으로 나누자
x_train = x[:7]
x_train = x[7:]
y_train = y[:7]
y_train = y[7:]


print(x_train, y_train)
