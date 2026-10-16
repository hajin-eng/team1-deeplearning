"""09. HGB 설정 조정 + 환경 열 포함/제외 비교 (트리 모델)

[1] HGB(HistGradientBoosting) 설정 8가지를 쌍둥이 그룹 분할로 비교한다.
    설정을 고른 뒤 '그 설정이 가장 좋아 보이는 건 고르는 과정에서 생긴 착시일 수 있다'.
    그래서 가장 좋은 설정과 기준 설정을 시드를 바꿔서 한 번 더 잰다(선택 편향 확인).

[2] 환경 열(공장 온도·습도, 공기압, 냉각수)을 포함/제외해서 RF, ExtraTrees, HGB를 비교한다.

이 실험이 필요한 이유
  - [1] 설정을 조정하면 점수가 얼마나 오르는지, 그 향상이 진짜인지 확인한다.
  - [2] 06번에서 본 '환경 열이 시계 역할을 한다'는 경향이 RF 말고 다른 트리 모델에서도 같은지 확인한다.

시간순 점수는 검증 구간에 있는 클래스만 평균한 방식이다.
시간이 오래 걸린다(10분 이상). 빠르게 확인하려면 --quick.
실행: python analysis/09_hgb_tuning_and_env.py
"""


# 실행 결과 요약
# HGB 설정 8가지 중 그룹 분할 최고는 B(0.310)로 기준 A(0.275)보다 +0.035 높았지만 고르는 과정의 착시가 섞여 있음
# 새 시드로 다시 재면 A 0.287, B 0.303으로 차이가 +0.016까지 줄어, 설정 조정의 실제 이득은 +0.01~0.02 정도
# 환경 열 6개를 빼면 RF 0.227 -> 0.160, ExtraTrees 0.245 -> 0.157, HGB 0.275 -> 0.233으로 모두 떨어지고, 시간순은 거의 변화 없음
# 06번의 RF 결과(환경 열이 시계 역할)가 ExtraTrees·HGB에서도 같은 방향이며, HGB는 하락 폭이 가장 작음
# 위 숫자는 빠른 모드(시드 1개) 값이라 전체 모드에서는 조금 달라질 수 있음 (이전 전체 모드: 최고 0.306, 새 시드 기준 대비 +0.009)

# 코드 메모
# def hgb(**options): 정의에 쓴 **는 호출할 때 받은 '이름=값' 인자들을 딕셔너리 하나로 모음
#   예) hgb(max_iter=100, learning_rate=0.05) -> options == {"max_iter": 100, "learning_rate": 0.05}
# HistGradientBoostingClassifier(**params): 호출에 쓴 **는 반대로 딕셔너리를 풀어서 인자로 넘김
# 설정마다 바꾸는 항목이 달라서 바꿀 것만 골라 적을 수 있게 이렇게 받음

import common
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from preprocess import ENV_COLS, load_data

common.announce("09 HGB 설정 조정, 환경 열 비교", "10분 이상", "4분")

# 선택 편향 확인에 쓰는, 설정을 고를 때 쓰지 않은 시드
if common.QUICK:
    RECHECK_SEEDS = [10]
else:
    RECHECK_SEEDS = [10, 11, 12]


def hgb(**options):
    """HGB를 만드는 함수를 돌려준다. 기본값(class_weight 등)에 options를 덮어쓴다."""
    # 기본값에 options를 덮어씀 (**의 뜻은 위의 코드 메모 참고)
    params = dict(class_weight="balanced", random_state=0)
    params.update(options)  # 설정마다 달라지는 값만 options로 넘기면 됨

    def make_model():
        return HistGradientBoostingClassifier(**params)

    return make_model


data = load_data()
X = data["X_train"].values
y = data["y_train"].values
groups = data["groups_train"].values

# ===============================================================
# 1. HGB 설정 조정
#    max_depth/max_leaf_nodes: 트리 크기 (작을수록 단순)
#    min_samples_leaf: 잎 하나에 들어갈 최소 행 수 (클수록 단순)
#    l2_regularization: 규제 세기, learning_rate x max_iter: 학습 양
#    목적: 단순한 설정 쪽(규제를 강하게)이 더 잘 일반화하는지 봄.
#          쌍둥이 그룹 분할 점수가 기준 A(0.283)보다 실제로 오르는지 확인하는 것이 핵심임.
# ===============================================================
GRID = {
    "A 기준(lr.08, 150회, 리프31)": hgb(max_iter=150, learning_rate=0.08),
    "B 리프15, lr.05, 100회": hgb(max_iter=100, learning_rate=0.05, max_leaf_nodes=15),
    "C 깊이3, lr.05, 200회": hgb(max_iter=200, learning_rate=0.05, max_depth=3),
    "D 깊이3, 리프최소50": hgb(
        max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=50
    ),
    "E 깊이4, 최소50, L2=1, lr.03, 300회": hgb(
        max_iter=300,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=50,
        l2_regularization=1.0,
    ),
    "F 깊이2, lr.05, 100회": hgb(max_iter=100, learning_rate=0.05, max_depth=2),
    "G 리프8, 최소100, lr.1, 50회": hgb(
        max_iter=50, learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=100
    ),
    "H 깊이5, L2=5, lr.05, 400회": hgb(
        max_iter=400, learning_rate=0.05, max_depth=5, l2_regularization=5.0
    ),
}

print("######## 1. HGB 설정 조정 (쌍둥이 그룹 분할 5-fold x 시드) ########", flush=True)
group_scores = {}
for name, make_model in GRID.items():
    scores = common.group_cv_scores(make_model, X, y, groups)
    group_scores[name] = scores
    time_score = common.time_split_score(make_model, X, y)
    print(
        f"  {name:36s} 그룹 F1 {common.fmt(scores)} | 시간순 {time_score:.3f}",
        flush=True,
    )

baseline_name = list(GRID)[0]
# 평균 점수가 가장 높은 설정을 찾음. 같으면 먼저 나온 것을 유지하려고 '>'로만 바꿈.
best_name = baseline_name
for config_name, config_scores in group_scores.items():
    if config_scores.mean() > group_scores[best_name].mean():
        best_name = config_name
gain = group_scores[best_name].mean() - group_scores[baseline_name].mean()
print(f"\n  >> 1차 최고: {best_name} (기준 대비 {gain:+.3f})", flush=True)

# 선택 편향 확인: 8개 중 가장 좋은 것을 고르면, 운이 좋아서 높게 나온 것이 뽑히기 쉬움.
# 설정을 고르는 데 쓰지 않은 시드로 다시 재서 향상이 유지되는지 봄.
print(
    f"  선택 편향 확인: 시드를 바꿔({RECHECK_SEEDS}) 기준과 최고를 다시 잼", flush=True
)
for name in dict.fromkeys([baseline_name, best_name]):  # 중복 없이 순서 유지
    scores = common.group_cv_scores(GRID[name], X, y, groups, seeds=RECHECK_SEEDS)
    print(f"    {name:36s} 새 시드 그룹 F1 {common.fmt(scores)}", flush=True)

# ===============================================================
# 2. 환경 열 포함 vs 제외
#    목적: 환경 열이 점수를 올리는 이유가 '시기를 알려 주는 시계'라면,
#          그룹 분할에서는 점수가 떨어지고 시간순에서는 차이가 작아야 함.
#    '제외가 나은 횟수'가 0에 가까우면 환경 열이 일관되게 도움이 된다는 뜻임.
# ===============================================================
print("\n######## 2. 환경 열 포함 vs 제외 (그룹 분할 + 시간순) ########", flush=True)
X_no_env = load_data(drop_cols=ENV_COLS)["X_train"].values


def make_rf():
    return RandomForestClassifier(
        300, class_weight="balanced_subsample", n_jobs=-1, random_state=0
    )


def make_extra_trees():
    return ExtraTreesClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0)


TREE_MODELS = {
    "RandomForest": make_rf,
    "ExtraTrees": make_extra_trees,
    "HGB(기준 A)": GRID[baseline_name],
}

for name, make_model in TREE_MODELS.items():
    with_env = common.group_cv_scores(make_model, X, y, groups)
    without_env = common.group_cv_scores(make_model, X_no_env, y, groups)
    diff = without_env - with_env  # 음수면 환경 열을 뺐을 때 점수가 떨어진 것

    time_with = common.time_split_score(make_model, X, y)
    time_without = common.time_split_score(make_model, X_no_env, y)

    print(
        f"  {name:14s} 포함 {with_env.mean():.3f} / 제외 {without_env.mean():.3f}",
        f"(차이 {diff.mean():+.3f}, 제외가 나은 횟수 {int((diff > 0).sum())}/{len(diff)})",
        f"| 시간순 포함 {time_with:.3f} / 제외 {time_without:.3f}",
        flush=True,
    )
