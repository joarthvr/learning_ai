import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

df = pd.DataFrame(
    {
        'x1': [1, 2, 3, 4, 5],
        'x2': [5, 4, 3, 2, 1],
        'x3': [-1000, 50, 7, 100, -1100],
        'Y': [2, 3, 4, 5, 6],
    }
)

print(df)
#! 5. 상관관계 히트맵 ------------------------------------------------
# 상관행렬은 모델이 아니라 데이터(DataFrame)로 계산한다
print(df.corr())
# ? x1 - Y : +1  x1 이 1 늘면 Y 도 항상 1 늘어난다 (완전한 양의 상관)
# ? x2 - Y : -1  x2 가 1 줄면 Y 는 1 늘어난다 (완전한 음의 상관)
# ? x1 - x2: -1  서로 같은 정보라 하나만 있어도 된다 (다중공선성)
# ? x3 - Y : 거의 0  값의 크기가 커도 Y 와 직선 관계가 없으면 상관은 낮다
# ? → 상관계수는 값의 크기(단위, 스케일)에 영향받지 않고, 함께 움직이는 정도만 본다

sns.heatmap(
    data=df.corr(),
    square=True,
    annot=True,  # 칸에 숫자 표시
    cbar=True,  # 색상 막대
    vmin=-1,  # 색 범위를 상관계수 범위로 고정
    vmax=1,
    cmap='coolwarm',  # 음수는 파랑, 양수는 빨강
)
plt.tight_layout()  # 긴 컬럼 이름이 잘리지 않게
plt.show()
