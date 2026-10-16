"""06. 환경 열이 '시기를 알려 주는 시계' 역할을 하는가 (RF)

환경 열 = 공장 온도·습도, 공기압, 냉각수 온도·압력(센서 5개) + 공장 결측 표시 열(factory_missing).
센서 값은 시기에 따라 천천히 변한다.
모델이 환경 열의 값으로 '지금이 언제인지'를 알아내서 맞히는 것이라면,
  - 그룹 분할(시기가 섞인 검증)에서는 점수가 오르지만
  - 시간순 분할(미래를 맞히는 검증)에서는 도움이 되지 않을 것이다.

열을 빼 보고 두 분할에서 점수가 어떻게 변하는지 본다.

이 실험이 필요한 이유: 환경 열을 전처리에서 유지할지 정해야 하는데,
환경 열이 '진짜 공정 정보'인지 '시기를 알려 주는 표지'인지에 따라 해석이 달라진다.
(공장 습도 Factory_Humidity는 04번 실험에서 RF 변수 중요도 1위였다)

시간순 분할은 처음 실험과 같은 값을 재현하려고 '모든 클래스 평균' 방식을 쓴다.
실행: python analysis/06_env_columns_drop.py    (--quick: 시드 1개)
"""


# 실행 결과 요약
# 환경 열 6개를 모두 빼면 쌍둥이 그룹 분할 점수는 0.231 -> 0.159로 떨어지고, 시간순 80/20은 0.219 -> 0.216으로 거의 변화 없음
# 센서 5개(공장 온도·습도, 공기압, 냉각수 온도·압력)가 공정 정보라기보다 시기를 알려 주는 표지(시계) 역할을 하는 것으로 추정됨
# 리더보드(랜덤 분할)에서는 도움이 되지만 실제 배포 성능과는 다를 수 있어 환경 열은 유지하고, 모델 비교 때 포함/제외 두 버전을 모두 측정함

# 코드 메모
# from preprocess import ENV_COLS
# ENV_COLS는 preprocess.py에 정의된 환경 열 6개의 이름 목록
# (Factory_Temp, Factory_Humidity, factory_missing, Air_Pressure, Coolant_Temp, Coolant_Pressure)
# 06·07·09번이 같은 목록을 쓰도록 한 곳에서 정의해 두고 import함
# drop(columns=...)에는 튜플을 그대로 넘기면 열 이름 하나로 해석되므로 list()로 바꿔서 사용함

import common
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from preprocess import ENV_COLS, load_data

common.announce("06 환경 열 제외 실험", "1분", "20초")


def make_rf():
    return RandomForestClassifier(
        300, class_weight="balanced_subsample", n_jobs=-1, random_state=0
    )


data = load_data()
X_all = data["X_train"]  # DataFrame 그대로 둠 (열 이름으로 열을 뺄 수 있게)
y = data["y_train"].values
groups = data["groups_train"].values

# 빼 볼 열 묶음. 환경 열을 한꺼번에 빼면 효과가 섞이므로 두 단계로 나눠 봄.
FEATURE_SETS = {
    "전부(25개)": [],
    "공장 온도·습도 제외": ["Factory_Temp", "Factory_Humidity", "factory_missing"],
    "공장+공기압+냉각수 제외": list(ENV_COLS),
}

for name, drop in FEATURE_SETS.items():
    X = X_all.drop(columns=drop).values

    # 두 분할의 점수가 서로 다르게 움직이는지가 핵심 질문.
    group_scores = common.group_cv_scores(make_rf, X, y, groups)
    time_score = common.time_split_score(make_rf, X, y, present_only=False)

    print(
        f"{name:22s} 그룹 분할 {common.fmt(group_scores)} | 시간순 80/20 {time_score:.3f}",
        flush=True,
    )
