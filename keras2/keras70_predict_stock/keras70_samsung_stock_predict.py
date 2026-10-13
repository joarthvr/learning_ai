import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

PATH = './_data/samsung/'
PATH_SAVE = './_save/keras70/'

# ====================================================================
# 상수
# ====================================================================
TIMESTEPS = 20  # 윈도우 길이: 최근 20거래일로 다음 날 종가를 예측
TEST_RATIO = 0.2  # 시간순으로 뒤쪽 20% 를 test

# 시가 고가 저가 종가 거래량 개인 기관 외국인 프로그램 외인비
USE_COLS = [1, 2, 3, 4, 8, 11, 12, 14, 15, 16]
CLOSE_IDX = 3  # USE_COLS 안에서 종가의 위치
FILES = {
    'x1_samsung': '삼성전자261012.csv',
    'x2_hynix': '하이닉스261012.csv',
    'x3_amore': '아모레261012.csv',
}


#! 1. data -------------------------------------------------------------
def load(filename):
    df = pd.read_csv(PATH + filename, encoding='cp949', thousands=',')
    out = df.iloc[:, USE_COLS].astype(float)
    out.index = pd.to_datetime(df.iloc[:, 0], format='%Y/%m/%d')
    return out.sort_index()


raw = {name: load(f) for name, f in FILES.items()}

# 세 종목 모두 거래한 날짜만 남김 (삼성이 가장 짧아서 2022-03-14~)
common = raw['x1_samsung'].index
for df in raw.values():
    common = common.intersection(df.index)
raw = {name: df.loc[common].to_numpy() for name, df in raw.items()}
print('공통 거래일:', len(common), common[0].date(), '~', common[-1].date())

# 시간순 분할 기준(행 번호). 스케일러는 이 앞쪽(train 구간)으로만 fit
n_rows = len(common)
split_row = int(n_rows * (1 - TEST_RATIO))

scalers = {}
scaled = {}
for name, arr in raw.items():
    scalers[name] = StandardScaler().fit(arr[:split_row])
    scaled[name] = scalers[name].transform(arr)

# 타깃: 삼성 종가. 이것도 train 구간으로만 fit
# 가격(원)을 직접 맞추면 train 범위 밖 가격을 못 맞히므로, 다음 날 등락률을 맞추고 종가로 환산
close = raw['x1_samsung'][:, CLOSE_IDX]
ret = close[1:] / close[:-1] - 1  # ret[k] = 행 k+1 의 전일 대비 등락률
scaler_y = StandardScaler().fit(ret[: split_row - 1].reshape(-1, 1))
ret_scaled = scaler_y.transform(ret.reshape(-1, 1))


def split_windows(data, timesteps):
    # i 번째 윈도우 = data[i : i+timesteps]  (행 i .. i+timesteps-1)
    return np.array([data[i : i + timesteps] for i in range(len(data) - timesteps + 1)])


# 윈도우 i 의 라벨 = 윈도우 바로 다음 날(행 i+timesteps) 종가
win = {name: split_windows(arr, TIMESTEPS) for name, arr in scaled.items()}
n_win = n_rows - TIMESTEPS  # 라벨이 있는 윈도우 수 (마지막 윈도우는 라벨 없음)
y_all = ret_scaled[TIMESTEPS - 1 :]  # (n_win, 1) 윈도우 다음 날 등락률

# 마지막 윈도우(마지막 거래일로 끝남) = "다음 날 종가 예측"에 쓸 입력
x1_pred = win['x1_samsung'][-1:]
x2_pred = win['x2_hynix'][-1:]
x3_pred = win['x3_amore'][-1:]

# 시간순 분할: 라벨 행(i+TIMESTEPS)이 split_row 앞이면 train, 이후면 test
n_train = split_row - TIMESTEPS
x1_train, x1_test = win['x1_samsung'][:n_train], win['x1_samsung'][n_train:n_win]
x2_train, x2_test = win['x2_hynix'][:n_train], win['x2_hynix'][n_train:n_win]
x3_train, x3_test = win['x3_amore'][:n_train], win['x3_amore'][n_train:n_win]
y_train, y_test = y_all[:n_train], y_all[n_train:]

print('x1', x1_train.shape, x1_test.shape)
print('x2', x2_train.shape, x2_test.shape)
print('x3', x3_train.shape, x3_test.shape)
print('y ', y_train.shape, y_test.shape)
print('pred 입력', x1_pred.shape, x2_pred.shape, x3_pred.shape)

#! 2~3. 모델 구성 / 저장된 모델을 그대로 불러온다
##################### 저장된 체크포인트 중 최신 파일 찾기 #####################
files = glob.glob(PATH_SAVE + 'k70_samsung-*.keras')
if not files:
    raise FileNotFoundError(
        f'체크포인트가 없습니다: {PATH_SAVE}\n-> keras70_samsung_stock.py 를 먼저 실행하세요.'
    )

best_path = max(files, key=os.path.getmtime)  # 수정 시각이 가장 늦은 것
print('')
print(f'불러온 파일: {os.path.basename(best_path)}  (후보 {len(files)}개 중)')
print('')
##############################################################################

model = load_model(best_path)
model.summary()

#! 4. evaluate, predict -------------------------------------------------------------
# 예측 등락률 → 종가: 윈도우 마지막 날 종가 * (1 + 등락률)
prev_close = close[TIMESTEPS - 1 : n_rows - 1]  # 윈도우 i 의 마지막 날 종가
true_close = close[TIMESTEPS:]  # 윈도우 i 의 다음 날 종가


def to_price(pred_scaled, prev):
    return prev * (1 + scaler_y.inverse_transform(pred_scaled).flatten())


# test 전체의 종가 오차
test_prev = prev_close[n_train:]
test_true = true_close[n_train:]
test_pred = to_price(model.predict([x1_test, x2_test, x3_test]), test_prev)
print('test MAE(원): ', np.abs(test_pred - test_true).mean())
print('기준선(전날 종가 그대로) MAE(원): ', np.abs(test_prev - test_true).mean())
print('마지막 2개 예측값: ', test_pred[-2:])
print('마지막 2개 정답: ', test_true[-2:])

next_close = to_price(model.predict([x1_pred, x2_pred, x3_pred]), close[-1])
print('다음 거래일 삼성전자 종가 예측: ', next_close)
