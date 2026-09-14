import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---  (저장할 때와 같은 값이어야 기록이 맞는다)
HIDDEN_UNITS = [64, 32, 16]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
PATH_SAVE = './_save/keras31/santander/'
SUBJECT = 'santander'  # 저장할 때 쓴 값과 반드시 동일해야 함
PREFIX = 'k31_santander'  # 체크포인트 파일명 앞부분
# ======================================================================

# 1. 데이터
DATA_PATH = './_data/kaggle_santander/'

train_csv = pd.read_csv(DATA_PATH + 'train.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y,  # 클래스 비율 유지
)

# ======================================================================
# 스케일러도 저장할 때와 같은 것을 써야 한다.
# 모델은 RobustScaler 로 변환된 값에 맞춰 학습됐으므로 다른 스케일러를 쓰면 엉뚱한 예측이 나온다
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# ======================================================================

# 2~3. 모델 구성 / 훈련 -> 하지 않는다. 저장된 모델을 그대로 불러온다
##################### 저장된 체크포인트 중 최신 파일 찾기 #####################
# save_best_only=True 라서 '가장 마지막에 저장된 파일' = 'val_loss 가 가장 낮은 모델'
files = glob.glob(PATH_SAVE + PREFIX + '*.keras')
if not files:
    raise FileNotFoundError(
        f'{SUBJECT} 체크포인트가 없습니다: {PATH_SAVE}\n'
        f'-> keras31_MCP_save_07_santander.py 를 먼저 실행하세요.'
    )

best_path = max(files, key=os.path.getmtime)  # 수정 시각이 가장 늦은 것
print('')
print(f'불러온 파일: {os.path.basename(best_path)}  (후보 {len(files)}개 중)')
print('')
##############################################################################

model = load_model(best_path)
model.summary()

# 4. 평가, 예측
y_prob = model.predict(x_test)
y_predict = np.round(y_prob)  # 0.5 기준 -> 0 또는 1
acc = accuracy_score(y_test, y_predict)

# target 이 90:10 이라 acc 는 전부 0으로 찍어도 0.90 -> AUC 를 같이 볼 것
auc = roc_auc_score(y_test, y_prob)

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(map(str, HIDDEN_UNITS))

print('')
print('===== RESULT (load) =====')
print(
    f'| subject={SUBJECT} | seed={SEED} | units={structure} | act={ACTIVATION} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| acc={acc:.4f} | auc={auc:.4f} '
    f'| file={os.path.basename(best_path)} |'
)
print('===================================')
print('')


"""
======================= 실험 기록 =======================

"""
