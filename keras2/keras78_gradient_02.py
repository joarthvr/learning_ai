X = -100000.0
EPOCHS = 100
LEARNING_RATE = 0.1


def f(x):
    return x**2 - 4 * x + 6


def gradient(x):
    return 2 * x - 4


y = f(X)
print('epoch\t x \t f(x)')
print('{:02d}\t {:6.5f}\t {:6.5f}\t'.format(0, X, f(X)))

x = X  # 시작점 복사. 루프 안에서 x 를 계속 갱신한다 (X 는 초기값 상수로 유지)

for i in range(EPOCHS):
    x = x - LEARNING_RATE * gradient(x)  # 이전 x 에서 출발해야 한 걸음씩 최솟값으로 이동
    print('{:02d}\t {:6.5f}\t {:6.5f}\t'.format(i + 1, x, f(x)))
