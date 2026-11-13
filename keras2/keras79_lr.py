X = 10
Y = 10
W = 0.001
LR = 0.999
EPOCHS = 100


for i, v in enumerate(range(EPOCHS)):
    hypothesis = X * W
    loss = (hypothesis - Y) ** 2  # ? mse, 바이어스, 시그마 신경 안씀

    print(i, 'Loss: ', round(loss, 4), '\tPredict: ', round(hypothesis, 4))

    up_predict = X * (W + LR)
    up_loss = (Y - up_predict) ** 2

    down_predict = X * (W - LR)
    down_loss = (Y - down_predict) ** 2
    # 소문자 w 에 복사해서 갱신하면 W 는 그대로라 매 epoch 같은 값에서 출발한다
    if up_loss > down_loss:
        W = W - LR
    else:
        W = W + LR
