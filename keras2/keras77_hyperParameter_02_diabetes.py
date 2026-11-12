import time
import warnings

from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

warnings.filterwarnings('ignore')
N_SPLITS = 5


es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=8,
    verbose=1,
    factor=0.8,
)

#! 1. DATA ---------------------------------------------------------------
x, y = load_diabetes(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8,
)
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=333)


#! 2. 모델 설정 -----------------------------------------------------------


def build_model(
    drop=0.5,
    optimizer='adam',
    activation='relu',
    node1=128,
    node2=64,
    node3=32,
    node4=16,
    node5=8,
    lr=0.001,
):
    # 탐색 공간의 node1~node5 이름과 인자 이름이 같아야 RandomizedSearchCV 가 넘겨줄 수 있다
    nodes = [node1, node2, node3, node4, node5]
    inputs = Input(
        shape=(10,),
        name='inputs',
    )

    for i, node in enumerate(nodes):
        if i == 0:
            x = Dense(
                node,
                activation=activation,
                name=f'hidden{i}',
            )(inputs)
        else:
            x = Dense(
                node,
                activation=activation,
                name=f'hidden{i}',
            )(x)
        x = Dropout(drop)(x)

    outputs = Dense(
        1,
        activation='linear',
        name='outputs',
    )(x)

    model = Model(inputs=inputs, outputs=outputs)

    #! 3. 컴파일 -----------------------------------------------------------
    model.compile(
        optimizer=optimizer,
        metrics=['mae'],
        loss='mse',
    )
    return model


def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16]
    return {
        'batch_size': batchs,
        'optimizer': optimizers,
        'drop': dropouts,
        'activation': activations,
        'node1': node1,
        'node2': node2,
        'node3': node3,
        'node4': node4,
        'node5': node5,
    }


hyperparameters = create_hyperparameter()


#! Keras 모델은 sklearn 추정기가 아니므로 KerasRegressor 로 감싼다
keras_model = KerasRegressor(
    build_fn=build_model,
    verbose=1,
)

# n_iter 는 정수 (True 는 1 로 해석돼 조합을 1번만 시도함)
model = RandomizedSearchCV(
    keras_model,
    hyperparameters,
    cv=3,
    verbose=1,
    n_iter=10,
    n_jobs=2,
)

#! 3. Fit
start = time.perf_counter()

model.fit(
    x_train,
    y_train,
    epochs=100,
    verbose=1,
    callbacks=[es, rlr],
    validation_split=0.1,
)
end = time.perf_counter()

print('최적의 매개변수: ', model.best_estimator_)
print('최적의 파라미터: ', model.best_params_)

#! 4. 평가 예측
print('model.r2: ', model.score(x_test, y_test))
print('best_r2: ', model.best_score_)

print('걸린시간:', round(end - start, 2), '초')

"""
model.r2:  -4192.748046875
best_r2:  -5401.777180989583
걸린시간: 35.42 초
"""
