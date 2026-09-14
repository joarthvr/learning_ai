import glob
import os

from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.15

# --- 모델 ---  (저장할 때와 같은 값이어야 기록이 맞는다)
HIDDEN_UNITS = [64, 32, 16, 8]
ACTIVATION = 'relu'

# --- 훈련 ---
EPOCHS = 1000
BATCH_SIZE = 4
LEARNING_RATE = 0.001
PATIENCE = 20

set_random_seed(SEED)
PATH_SAVE = './_save/keras31/boston/'
SUBJECT = 'boston'  # 저장할 때 쓴 값과 반드시 동일해야 함
# save 파일이 2회차(스케일링X / RobustScaler)를 돌리므로 회차까지 지정해야 한다.
# 여기서는 스케일링을 적용한 회차를 평가한다
PREFIX = 'k31_boston_RobustScaler'
# ======================================================================

# 1. 데이터
# boston_housing 은 이미 train/test 로 나뉘어 나오므로 train_test_split 이 없다
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(
    test_split=1 - TRAIN_SIZE,
    seed=SEED,
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
        f'-> keras31_MCP_save_03_boston.py 를 먼저 실행하세요.'
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


"""
