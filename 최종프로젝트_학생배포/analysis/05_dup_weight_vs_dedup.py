"""05. 중복 행 처리 비교: 그대로 vs 가중치 vs 그룹당 1행

세 가지 학습 방식
  all    : 모든 행을 그대로 학습
  weight : 같은 입력이 두 번 세지지 않도록 가중치 1/그룹 크기를 준다
  dedup  : 학습 데이터에서 그룹당 첫 행만 남긴다

이 비교가 필요한 이유: 입력이 같은 행이 3,174개(42%)라서, 이를 어떻게 다룰지가 전처리의 큰 결정이다.
지우면 정보를 버리고, 그대로 두면 같은 입력이 학습에 두 번 들어간다.

검증은 항상 쌍둥이 그룹 분할이고, 검증 행은 모두 쓴다 (test에도 중복 행이 있으므로).
'all 대비 차이'와 '나아진 횟수'로 일관되게 나아지는지 본다.
  - all 대비 차이: 같은 폴드에서 (weight 또는 dedup 점수 - all 점수)의 평균
  - 나아진 횟수: 폴드 5개 중 weight 또는 dedup이 all보다 점수가 높았던 폴드 수 (3/5 = 5개 중 3개)
  -> 차이가 표준편차(약 0.03)보다 작고 나아진 횟수가 절반 근처면 노이즈다.

시간순 분할은 처음 실험과 같은 값을 재현하려고 '모든 클래스 평균' 방식을 쓴다.
실행: python analysis/05_dup_weight_vs_dedup.py    (--quick: 시드 1개)
"""


# 실행 결과 요약
# weight·dedup 방식의 all 대비 점수 차이는 RF -0.001~+0.004, HGB +0.003~+0.013으로 표준편차(0.02~0.04)보다 작거나 비슷함
# 나아진 횟수(5개 폴드 중 all보다 점수가 높았던 폴드 수)는 RF 3/5·3/5, HGB 3/5·4/5로 절반 근처라 어느 방식도 일관되게 낫다고 보기 어려움
# HGB의 그룹당 1행(dedup)이 all 대비 +0.013(4/5)으로 가장 높지만 표준편차 안이고 시드 1개 결과라 확정할 수 없음
# 시간순 80/20은 한 번뿐이라 참고용: RF의 그룹당 1행(dedup)만 0.292로 높게 나왔으나 HGB에서는 재현되지 않아 노이즈로 판단함
# 따라서 행을 그대로 유지함 (가장 단순하고 test에도 같은 중복 구조가 있음)

# 코드 메모
# HistGradientBoostingClassifier (HGB)
# scikit-learn의 그래디언트 부스팅 분류 모델
# 얕은 트리를 순서대로 만들면서 새 트리가 앞 트리의 오류를 보완하도록 학습
# Hist: 연속값을 최대 255개 구간으로 묶어 계산해서 빠르고 결측값(NaN)도 자체적으로 처리함
# RF와 방식이 다른 모델로도 결과가 같은지 확인하려고 사용함

import common
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from preprocess import load_data

common.announce("05 중복 행 처리 비교", "4분", "1~2분")

data = load_data()
X = data["X_train"].values
y = data["y_train"].values
groups = data["groups_train"].values

# [가중치 준비] 가중치 = 1 / (그 행이 속한 그룹의 행 수)
# 2행 그룹이면 각 행이 0.5, 4행 그룹이면 각 행이 0.25 -> 그룹 전체가 1번만 세어짐.
group_size = pd.Series(groups).map(pd.Series(groups).value_counts()).values
weight = 1.0 / group_size


# 목적: 모델 종류에 따라 결과가 일관되게 나오는지 확인하기 위해 대표 모델 2개(RF, HGB)로 비교함.
def make_rf():
    return RandomForestClassifier(
        300, class_weight="balanced_subsample", n_jobs=-1, random_state=0
    )


def make_hgb():
    return HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=150, learning_rate=0.08, random_state=0
    )


# MODELS의 값은 '모델을 새로 만들어 주는 함수'(폴드마다 새 모델로 학습하기 위해).
MODELS = {
    "RF": make_rf,
    "HGB": make_hgb,
}
VARIANTS = ["all", "weight", "dedup"]


def fit_and_score(make_model, variant, train_idx, valid_idx):
    """variant 방식으로 학습하고 검증 행에서 macro F1을 잰다."""
    use_idx = train_idx  # 기본: 학습 행을 그대로 씀

    if variant == "dedup":
        # np.unique의 return_index: 각 그룹이 처음 나온 위치. 정렬해서 원래 순서를 유지함.
        # -> 학습 데이터에서 그룹당 첫 행만 남김. (검증 행은 건드리지 않음)
        _, first = np.unique(groups[train_idx], return_index=True)
        use_idx = train_idx[np.sort(first)]

    model = make_model()
    if variant == "weight":
        # sample_weight: 행마다 학습에 반영되는 비중. 같은 입력이 두 번 세지지 않게 함.
        model.fit(X[use_idx], y[use_idx], sample_weight=weight[use_idx])
    else:
        model.fit(X[use_idx], y[use_idx])

    return common.macro_f1(y[valid_idx], model.predict(X[valid_idx]))


for model_name, make_model in MODELS.items():
    # 목적: 같은 폴드 분할에서 세 방식을 모두 재야 '같은 조건에서의 짝 비교'가 됨.
    scores = {}
    for v in VARIANTS:
        scores[v] = []
    for _, train_idx, valid_idx in common.group_folds(y, groups):
        for variant in VARIANTS:
            scores[variant].append(
                fit_and_score(make_model, variant, train_idx, valid_idx)
            )

    base = np.array(scores["all"])
    n = len(base)
    print(f"--- {model_name} 쌍둥이 그룹 분할 ({n}회) ---", flush=True)
    for variant in VARIANTS:
        s = np.array(scores[variant])
        diff = s - base  # 같은 폴드끼리 뺀 차이 (짝 비교)
        print(
            f"  {variant:7s} macro F1 평균 {s.mean():.3f} (표준편차 {s.std():.3f})",
            f"| all 대비 차이 평균 {diff.mean():+.3f}, 나아진 횟수 {int((diff > 0).sum())}/{n}",
            flush=True,
        )

    # 시간순 80/20은 한 번뿐이라 참고용
    train_idx, valid_idx = common.time_split(len(y))
    time_scores = {}
    for v in VARIANTS:
        time_scores[v] = round(fit_and_score(make_model, v, train_idx, valid_idx), 3)
    print("  시간순 80/20 한 번:", time_scores, flush=True)
