"""
RF 설정 탐색
main.py의 RF 설정(잎 최소, 질문마다 보는 열 비율, 나무 수, 가중치 강도)을 여러 조합으로 바꿔 끼우며
세 가지 점수를 같이 본다.
  - 그룹 CV: 쌍둥이를 같은 조각에 넣은 점수 (처음 보는 입력에서의 성능, 모델 선택 기준), 시드 4개
  - 랜덤 CV: 쌍둥이가 갈라지는 점수 (외운 효과가 섞임), 시드 2개
  - 리더보드풍: 랜덤 CV 예측을 "쌍둥이가 학습에 있는 행 58% : 없는 행 42%"가 되도록 가중해 다시 채점한 값
    (test의 약 58%가 train에 같은 입력이 있다는 가정의 어림이고, 정확한 예측은 아님)

실행 방법: 학생배포 폴더에서  python 실험_RF설정탐색.py   (약 8분)
제출 파일은 만들지 않는다 (비교만 한다)
"""

import contextlib
import io
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import main  # main.py를 불러온다 (main()은 실행되지 않는다)

GROUP_SEEDS = [42, 0, 1, 2]  # 그룹 CV 시드 (main.py의 분할과 같은 방식)
RANDOM_SEEDS = [42, 0]  # 랜덤 CV 시드
TWIN_SHARE = 0.58  # 리더보드풍 점수에서 "쌍둥이가 학습에 있는 행"의 비율 가정

# 설정 이름 → (잎 최소, 열 비율, 나무 수, 가중치 강도). C안(현재 main.py 기본값)은 "잎4 열0.5"
CONFIGS = {
    "A안 잎5 열sqrt": (5, "sqrt", 300, 0.5),
    "열 0.3": (5, 0.3, 300, 0.5),
    "열 0.5": (5, 0.5, 300, 0.5),
    "열 0.7": (5, 0.7, 300, 0.5),
    "열 1.0": (5, 1.0, 300, 0.5),
    "나무 600": (5, "sqrt", 600, 0.5),
    "나무 1000": (5, "sqrt", 1000, 0.5),
    "강도 0.6": (5, "sqrt", 300, 0.6),
    "강도 0.75 (D안)": (5, "sqrt", 300, 0.75),
    "잎8 열0.5": (8, 0.5, 300, 0.5),
    "잎5 열0.5 강도0.75": (5, 0.5, 300, 0.75),
    "C안 잎4 열0.5": (4, 0.5, 300, 0.5),
    "잎6 열0.7": (6, 0.7, 300, 0.5),
}


def apply(leaf, max_features, n_trees, power):
    """main.py의 설정값을 바꿔 끼운다 (파일이 아니라 실행 중 메모리 값만 바뀜)."""
    main.MODEL_NAME = "rf"
    main.RF_MIN_LEAF = leaf
    main.RF_MAX_FEATURES = max_features
    main.RF_N_TREES = n_trees
    main.CLASS_WEIGHT_POWER = power


def group_score(X, y, groups, seed):
    """main.py와 같은 그룹 5조각으로 합쳐서 채점한 macro F1."""
    main.SEED = seed
    main.set_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):  # 조각별 출력은 숨김
        folds = main.split_validation(y, groups)
        oof, _ = main.cross_validate(X, y, folds)
    return f1_score(y, oof, labels=main.LABELS, average="macro", zero_division=0)


def random_scores(X, y, groups, seed):
    """랜덤 5조각의 합쳐서 채점한 macro F1과, 리더보드풍 가중 macro F1."""
    main.SEED = seed
    main.set_seed(seed)
    oof = y.copy().astype(object)
    has_twin = np.zeros(len(y), dtype=bool)
    cv = StratifiedKFold(main.VALID_FOLDS, shuffle=True, random_state=seed)
    for fit_pos, valid_pos in cv.split(X, y):
        model = main.build_model(y.iloc[fit_pos])
        model.fit(X.iloc[fit_pos], y.iloc[fit_pos])
        oof.iloc[valid_pos] = model.predict(X.iloc[valid_pos])
        has_twin[valid_pos] = groups.iloc[valid_pos].isin(set(groups.iloc[fit_pos])).values

    plain = f1_score(y, oof, labels=main.LABELS, average="macro", zero_division=0)
    weight = np.where(
        has_twin,
        TWIN_SHARE / has_twin.sum(),
        (1 - TWIN_SHARE) / (~has_twin).sum(),
    ) * len(y)
    weighted = f1_score(
        y, oof, labels=main.LABELS, average="macro", zero_division=0, sample_weight=weight
    )
    return plain, weighted


start = time.time()
data = main.load_data(drop_cols=main.DROP_COLS)
main.check_data(data)
X, y, groups = data["X_train"], data["y_train"], data["groups_train"]

print(f"\n{'설정':22s} 그룹CV(4시드) ±표준편차  랜덤CV  리더보드풍(2시드)")
for name, (leaf, max_features, n_trees, power) in CONFIGS.items():
    apply(leaf, max_features, n_trees, power)
    group = [group_score(X, y, groups, s) for s in GROUP_SEEDS]
    rand = [random_scores(X, y, groups, s) for s in RANDOM_SEEDS]
    print(
        f"{name:22s} {np.mean(group):.4f}      ±{np.std(group):.4f}     "
        f"{np.mean([r[0] for r in rand]):.3f}   {np.mean([r[1] for r in rand]):.3f}"
        f"   ({time.time() - start:.0f}초)",
        flush=True,
    )

print("\n읽는 법: 그룹CV 차이가 ±0.01보다 작으면 노이즈. 리더보드풍이 높을수록 쌍둥이를 더 외우는 설정.")
