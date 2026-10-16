"""10. 남은 전처리 선택 비교: 이상치 행 삭제 / Cylinder_Pressure 제거 / 클래스 불균형 처리

[1] 이상치 행 삭제와 Cylinder_Pressure 제거를 'train에서만' 바꿔 보고, 기본 상태와 점수를 비교한다.
    - 이상치 행 = is_restart, bad_biscuit, long_cycle 중 하나라도 1인 행 (train 11행)
    - 검증 행은 항상 전부 쓴다 (test에도 이상치 행이 있으므로, 삭제의 영향을 검증에서 가리지 않기 위해).
    - 모델: RF, HGB (05번과 같은 설정)

[2] 클래스 불균형 처리 방식 4가지를 HGB로 비교한다.
    - 가중치 없음
    - balanced           : 클래스 크기에 반비례하는 가중치 (지금까지 쓴 기본값)
    - 제곱근 가중치       : 1 / sqrt(클래스 크기). balanced보다 약하게 소수 클래스를 키움
    - 오버샘플링          : 학습 폴드 안에서만 소수 클래스 행을 복제해 클래스 크기를 맞춤
    - 확률 보정           : 가중치 없이 학습한 뒤 예측 확률을 학습 데이터의 클래스 비율로 나눔

이 비교가 필요한 이유: 세 가지 모두 전처리에서 아직 정하지 못한 선택이다.
점수 차이가 표준편차(약 0.03)보다 작으면 어느 쪽이든 상관없다고 보고 가장 단순한 쪽을 유지한다.

검증은 쌍둥이 그룹 분할이다. 오버샘플링은 반드시 학습 폴드 안에서만 한다
(검증 폴드에 복제본이 들어가면 쌍둥이가 새는 것과 같은 문제가 생긴다).
실행: python analysis/10_remaining_choices.py    (--quick: 시드 1개)
"""


# 실행 결과 요약
# 이상치 11행 삭제: RF +0.001(9/15), HGB +0.006(10/15)로 표준편차(0.03) 안쪽이라 효과를 구분할 수 없음 -> 유지
# Cylinder_Pressure 제거: RF -0.003(7/15), HGB +0.008(10/15)로 모델에 따라 방향이 갈리고 표준편차 안쪽 -> 유지
# 클래스 불균형: 가중치 없음 -0.025, 제곱근 가중치 -0.015, 오버샘플링 +0.001은 balanced보다 낫지 않음
# 확률 보정만 +0.019(12/15)로 가장 높지만 불량 탐지율 30% -> 50%, 정상 헛경보율 13% -> 25%로 오르는 대가가 있음

import common
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from preprocess import OUTLIER_FLAGS, load_data

common.announce("10 남은 전처리 선택 비교", "8분", "2~3분")

data = load_data()
X_df = data["X_train"]
X = X_df.values
y = data["y_train"].values
groups = data["groups_train"].values

# 이상치 행 위치 (True면 이상치 행). 표시 열 3개 중 하나라도 1이면 이상치 행.
is_flagged = X_df[OUTLIER_FLAGS].sum(axis=1).values > 0

# Cylinder_Pressure를 뺀 열 위치 목록
all_cols = list(X_df.columns)
cols_without_cylinder = []
for i, c in enumerate(all_cols):
    if c != "Cylinder_Pressure":
        cols_without_cylinder.append(i)
cols_all = list(range(len(all_cols)))


def make_rf():
    return RandomForestClassifier(
        300, class_weight="balanced_subsample", n_jobs=-1, random_state=0
    )


def make_hgb(class_weight="balanced"):
    return HistGradientBoostingClassifier(
        class_weight=class_weight, max_iter=150, learning_rate=0.08, random_state=0
    )


def defect_rates(y_true, pred):
    """불량 탐지율(실제 불량 중 '정상이 아님'으로 예측한 비율)과 정상 헛경보율을 돌려준다."""
    is_normal = y_true == "정상"
    detect = ((pred != "정상") & ~is_normal).sum() / (~is_normal).sum()
    false_alarm = ((pred != "정상") & is_normal).sum() / is_normal.sum()
    return detect, false_alarm


# ===============================================================
# 1. 이상치 행 삭제 / Cylinder_Pressure 제거
#    목적: 둘 다 train에서만 바꾸고 검증 행은 그대로 둔 채 점수가 달라지는지 확인.
#    짝 비교: 같은 폴드에서 (바꾼 점수 - 기본 점수)를 구해 평균내고, 기본보다 나아진 폴드 수를 센다.
# ===============================================================
# (이름, 쓸 열 위치, 이상치 행을 train에서 뺄지)
VARIANTS = [
    ("기본(25개 열, 이상치 행 유지)", cols_all, False),
    ("이상치 11행 삭제", cols_all, True),
    ("Cylinder_Pressure 제거", cols_without_cylinder, False),
]
MODELS = {
    "RF": make_rf,
    "HGB": make_hgb,
}

print("######## 1. 이상치 행 삭제 / Cylinder_Pressure 제거 (쌍둥이 그룹 분할) ########")
part1 = {}
for model_name in MODELS:
    part1[model_name] = {}
    for variant_name, _, _ in VARIANTS:
        part1[model_name][variant_name] = []

for _, train_idx, valid_idx in common.group_folds(y, groups):
    for model_name, make_model in MODELS.items():
        for variant_name, use_cols, drop_flagged in VARIANTS:
            use_idx = train_idx
            if drop_flagged:
                # 이상치 행만 학습에서 뺀다. 검증 행(valid_idx)은 건드리지 않는다.
                use_idx = train_idx[~is_flagged[train_idx]]

            model = make_model()
            model.fit(X[use_idx][:, use_cols], y[use_idx])
            pred = model.predict(X[valid_idx][:, use_cols])
            score = common.macro_f1(y[valid_idx], pred)
            part1[model_name][variant_name].append(score)

for model_name in MODELS:
    print(f"--- {model_name} ---", flush=True)
    base = np.array(part1[model_name][VARIANTS[0][0]])
    for variant_name, _, _ in VARIANTS:
        s = np.array(part1[model_name][variant_name])
        diff = s - base
        print(
            f"  {variant_name:28s} macro F1 {common.fmt(s)}",
            f"| 기본 대비 {diff.mean():+.3f}, 나아진 횟수 {int((diff > 0).sum())}/{len(diff)}",
            flush=True,
        )

# ===============================================================
# 2. 클래스 불균형 처리 (HGB)
#    목적: 소수 클래스를 키우는 방식에 따라 점수와 '많이 잡고 많이 틀리는' 정도가 어떻게 달라지는지 봄.
# ===============================================================
IMBALANCE_NAMES = [
    "가중치 없음",
    "balanced (기준)",
    "제곱근 가중치",
    "오버샘플링",
    "확률 보정",
]


def sqrt_weights(labels):
    """클래스 크기의 제곱근에 반비례하는 행별 가중치 (평균이 1이 되도록 맞춤)."""
    classes, counts = np.unique(labels, return_counts=True)
    weight_of_class = {}
    for c, n in zip(classes, counts):
        weight_of_class[c] = 1.0 / np.sqrt(n)

    weights = np.zeros(len(labels))
    for i, label in enumerate(labels):
        weights[i] = weight_of_class[label]
    return weights / weights.mean()


def oversample(train_idx, seed):
    """학습 행 위치에서 소수 클래스 행을 복제해 모든 클래스를 가장 큰 클래스 크기로 맞춘다."""
    rng = np.random.RandomState(seed)
    labels = y[train_idx]
    classes, counts = np.unique(labels, return_counts=True)
    target = counts.max()

    picked = []
    for c in classes:
        members = train_idx[labels == c]
        picked.append(members)
        extra_count = target - len(members)
        if extra_count > 0:
            picked.append(rng.choice(members, size=extra_count, replace=True))
    return np.concatenate(picked)


print("\n######## 2. 클래스 불균형 처리 (HGB, 쌍둥이 그룹 분할) ########")
part2_f1 = {}
part2_detect = {}
part2_alarm = {}
for name in IMBALANCE_NAMES:
    part2_f1[name] = []
    part2_detect[name] = []
    part2_alarm[name] = []


def record(name, y_true, pred):
    part2_f1[name].append(common.macro_f1(y_true, pred))
    detect, alarm = defect_rates(y_true, pred)
    part2_detect[name].append(detect)
    part2_alarm[name].append(alarm)


for seed, train_idx, valid_idx in common.group_folds(y, groups):
    y_valid = y[valid_idx]

    # balanced (기준)
    model = make_hgb("balanced").fit(X[train_idx], y[train_idx])
    record("balanced (기준)", y_valid, model.predict(X[valid_idx]))

    # 가중치 없음 + 같은 모델의 확률 보정 (한 번 학습한 모델을 두 방식에 쓴다)
    model = make_hgb(None).fit(X[train_idx], y[train_idx])
    record("가중치 없음", y_valid, model.predict(X[valid_idx]))

    # 확률 보정: 확률 / 학습 데이터의 클래스 비율. 흔한 클래스는 낮추고 드문 클래스는 올림.
    ratio = np.zeros(len(model.classes_))
    for k, c in enumerate(model.classes_):
        ratio[k] = (y[train_idx] == c).mean()
    proba = model.predict_proba(X[valid_idx])
    corrected = model.classes_[np.argmax(proba / ratio, axis=1)]
    record("확률 보정", y_valid, corrected)

    # 제곱근 가중치
    weights = sqrt_weights(y[train_idx])
    model = make_hgb(None).fit(X[train_idx], y[train_idx], sample_weight=weights)
    record("제곱근 가중치", y_valid, model.predict(X[valid_idx]))

    # 오버샘플링 (학습 폴드 안에서만)
    over_idx = oversample(train_idx, seed)
    model = make_hgb(None).fit(X[over_idx], y[over_idx])
    record("오버샘플링", y_valid, model.predict(X[valid_idx]))

base = np.array(part2_f1["balanced (기준)"])
for name in IMBALANCE_NAMES:
    s = np.array(part2_f1[name])
    diff = s - base
    print(
        f"  {name:16s} macro F1 {common.fmt(s)}",
        f"| 기준 대비 {diff.mean():+.3f}, 나아진 횟수 {int((diff > 0).sum())}/{len(diff)}",
        f"| 탐지율 {np.mean(part2_detect[name]):.1%}, 헛경보율 {np.mean(part2_alarm[name]):.1%}",
        flush=True,
    )
