import glob
import os

from sklearn.datasets import load_diabetes
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 153

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.2

# --- 모델 ---
HIDDEN_UNITS = [3, 9, 3]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.001
PATIENCE = 15

set_random_seed(SEED)
PATH_SAVE = './_save/keras31/diabetes/'
SUBJECT = 'diabetes'
# ======================================================================

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

INPUT_DIM = x.shape[1]  # 10
OUTPUT_DIM = 1  # 회귀

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
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
files = glob.glob(PATH_SAVE + f'k31_{SUBJECT}_*.keras')
if not files:
    raise FileNotFoundError(
        f'{SUBJECT} 체크포인트가 없습니다: {PATH_SAVE}\n'
        f'-> keras31_MCP_save_02_diabetes.py 를 먼저 실행하세요.'
    )

best_path = max(files, key=os.path.getmtime)  # 수정 시각이 가장 늦은 것
print('')
print(f'불러온 파일: {os.path.basename(best_path)}  (후보 {len(files)}개 중)')
print('')
##############################################################################

model = load_model(best_path)
model.summary()

# 4. 평가, 예측
rmse = root_mean_squared_error(y_test, model.predict(x_test))

##################### 실험 결과 요약 (기록용) #####################
structure = '-'.join(map(str, HIDDEN_UNITS))

print('')
print('===== RESULT (load) =====')
print(
    f'| subject={SUBJECT} | seed={SEED} | units={structure} | act={ACTIVATION} '
    f'| ts={TRAIN_SIZE} | vs={VAL_SPLIT} | bs={BATCH_SIZE} | ep={EPOCHS} '
    f'| lr={LEARNING_RATE} | pat={PATIENCE} '
    f'|| rmse={rmse:.4f} '
    f'| file={os.path.basename(best_path)} |'
)
print('===================================')
print('')


"""
===== RESULT (load) =====
| subject=diabetes | seed=153 | units=3-9-3 | act=relu | ts=0.8 | vs=0.2 | bs=10 |ep=100 | lr=0.001 | pat=15
|| rmse=53.9068 | file=k31_diabetes_0914_1419-0100-3500.2659.keras |
===================================
"""
