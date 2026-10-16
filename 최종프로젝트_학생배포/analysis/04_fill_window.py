"""04. 결측 채우기 방식에 따라 점수와 예측이 달라지는가

비교하는 방식 (preprocess.load_data의 fill 옵션)
  - median        : train 중앙값
  - neighbor      : 결측 덩어리 앞뒤 실측값 window개씩의 중앙값 (window 10 / 30 / 60)
  - interp        : 앞뒤 샷 선형 보간

보는 것
  1) 방식별 macro F1 (RF, 쌍둥이 그룹 분할)
  2) 결측이 있는 Factory_Temp·Factory_Humidity 열이 RF에서 얼마나 중요한 변수인가
  3) 같은 모델로 test 결측 행을 예측했을 때 방식에 따라 답이 바뀌는가

이 점검이 필요한 이유: 결측 90행을 어떻게 채울지는 팀이 정해야 하는데,
방식에 따라 점수가 달라진다면 신중히 골라야 하고, 같다면 단순한 방식을 써도 된다.

읽는 법: 점수 차이가 표준편차(약 0.03)보다 작으면 방식 차이는 점수에 영향이 없다는 뜻이다.
실행: python analysis/04_fill_window.py    (--quick: 시드 1개)
"""


# 실행 결과 요약
# 결측 채우기 방식(전체 중앙값/앞뒤 실측값 중앙값 w=10·30·60/선형보간)에 따른 점수 차이가 거의 없음
# 같은 모델로 test 결측 27행을 예측하면 방식에 따라 예측이 바뀌는 행은 0~3행으로 파악됨
# 결측이 있는 Factory_Humidity는 RF 변수 중요도 1위, Factory_Temp는 4위
# 결측 표시 열(factory_missing)은 22위(0.001)로 거의 쓰이지 않음
# 따라서 채우는 방식은 점수에 영향이 없는 것으로 판단되어 앞뒤 실측값 중앙값 w=30(앞뒤 30개씩)을 선택함

# 코드 메모
# load_data(**options)
# 방식마다 넘기는 옵션의 개수와 이름이 다름
# median은 fill만, neighbor는 fill과 window
# 옵션을 딕셔너리에 담아 두고 **로 풀어서 같은 한 줄로 반복 호출 하는 것이 좋다고 판단함
# cf) 풀어 쓰면 기본값(window=30)을 호출하는 쪽에 다시 작성해야 해서 코드 길이 증가, 기본값 변경 시 어긋날 가능성 존재

import common
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data

common.announce("04 결측 채우기 방식 비교", "2분", "40초")


def make_rf():
    # 비교 대상은 '채우기 방식'이므로 모델과 시드는 항상 같게 고정한다.
    return RandomForestClassifier(
        300, class_weight="balanced_subsample", n_jobs=-1, random_state=0
    )


# (이름, load_data에 넘길 옵션) 목록. 방식만 바꿔 같은 코드를 반복하려고 목록으로 둠.
# 아래 load_data(**options)의 **는 딕셔너리를 '이름=값' 인자로 풀어 넘기는 문법.
#   예) {"fill": "neighbor", "window": 10} -> load_data(fill="neighbor", window=10)
FILLS = [
    ("median", dict(fill="median")),
    ("neighbor w=10", dict(fill="neighbor", window=10)),
    ("neighbor w=30", dict(fill="neighbor", window=30)),
    ("neighbor w=60", dict(fill="neighbor", window=60)),
    ("interp", dict(fill="interp")),
]

# ---------------------------------------------------------------
# 1. 방식별 점수
#    목적: 채우는 방식을 바꿨을 때 쌍둥이 그룹 분할 점수가 달라지는지 확인.
# ---------------------------------------------------------------
print("=== 1. 채우기 방식별 점수 (RF, 쌍둥이 그룹 분할) ===")
for name, options in FILLS:
    # 전처리를 options에 적힌 채우기 방식으로 다시 실행
    data = load_data(**options)
    X = data["X_train"].values
    y = data["y_train"].values
    groups = data["groups_train"].values

    scores = common.group_cv_scores(make_rf, X, y, groups)
    print(f"  {name:14s} macro F1 {common.fmt(scores)}", flush=True)

# ---------------------------------------------------------------
# 2. 변수 중요도 (기본 설정으로 전체 train에 학습)
#    목적: 결측이 있는 열(Factory_Temp, Factory_Humidity)이 모델에 얼마나 중요한지 알아야,
#         채우는 방식이 점수에 영향을 줄 수 있는지 가늠할 수 있음.
# ---------------------------------------------------------------
base = load_data()
model = make_rf().fit(base["X_train"].values, base["y_train"].values)

importance = pd.Series(model.feature_importances_, index=base["X_train"].columns)
importance = importance.sort_values(ascending=False)
ranking = list(importance.index)

print(f"\n=== 2. Factory_* 열의 중요도 순위 (RF, 전체 {len(importance)}개 중) ===")
for col in ["Factory_Temp", "Factory_Humidity", "factory_missing"]:
    print(f"  {col:17s} {ranking.index(col) + 1}위, 중요도 {importance[col]:.3f}")

# ---------------------------------------------------------------
# 3. test의 결측 행 예측이 채우기 방식에 따라 바뀌는가
#    목적: 점수는 평균이라 작은 변화가 묻힐 수 있음. 결측 행 27개의 예측이 실제로 바뀌는지 직접 셈.
#    같은 모델(2번에서 학습한 것)로 입력만 바꿔 예측해 기준(neighbor w=30)과 비교.
# ---------------------------------------------------------------
is_missing = (base["X_test"]["factory_missing"] == 1).values
reference_pred = model.predict(base["X_test"].values)

print(f"\n=== 3. test 결측 {int(is_missing.sum())}행의 예측이 방식에 따라 바뀌나 ===")
for name, options in FILLS:
    # 방식(options)만 바꿔 전처리를 다시 하고, test 입력만 꺼내 같은 모델로 예측.
    X_test = load_data(**options)["X_test"].values
    pred = model.predict(X_test)
    changed = pred != reference_pred
    print(
        f"  {name:14s} 기준과 예측이 다른 행: 결측 중 {int((changed & is_missing).sum())}개,",
        f"전체 {len(pred)}행 중 {int(changed.sum())}개",
    )
