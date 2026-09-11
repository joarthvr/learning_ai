# keras29_3 에서 저장한 '훈련된 모델'을 불러와서 훈련 없이 평가만
# -> keras29_3 의 rmse 와 같은 값이 나와야 한다
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import set_random_seed

# ======================================================================
SEED = 42

# --- 데이터 ---
TRAIN_SIZE = 0.8

PATH_SAVE = './_save/keras29/'

set_random_seed(SEED)
# ======================================================================

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True,
)

# 저장할 때 쓴 스케일러와 동일하게 맞춰야 한다 (스케일러는 모델에 저장되지 않음)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성 + 3. 컴파일 훈련 -> 저장된 모델 불러오기로 전부 대체
model = load_model(PATH_SAVE + 'keras29_3_save_model2.keras')
model.summary()

# 4. 결과 예측 (훈련 X)
loss = model.evaluate(x_test, y_test, verbose=0)
rmse = root_mean_squared_error(y_test, model.predict(x_test))

print('')
print('===== RESULT (load only) =====')
print(f'| loss(mse)={loss:.4f} | rmse={rmse:.4f} |')
print('===================================')
print('')

"""
===== RESULT (load only) =====
| loss(mse)= | rmse= |
===================================
"""
