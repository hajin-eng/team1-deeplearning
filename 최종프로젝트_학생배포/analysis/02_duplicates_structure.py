"""02. 입력이 같은 행(쌍둥이)의 구조 조사

묻는 것: 왜 입력이 똑같은 행이 있을까?
  1) Shot 번호가 같은 행과, 입력이 같은 행은 같은 것인가?
  2) 우연의 일치일 수 있나? (입력이 다른 이웃 행은 얼마나 다른가)
  3) Shot 번호는 입력이 같은 행 묶음 사이에서 어떻게 올라가나?
  4) 입력이 같은 쌍(2행 묶음)의 두 행은 원본 불량 기록도 같은가?

결론은 확정이 아니라 단서다. 정리는 전처리_결정_정리.md의 '4. 중복 행 처리'를 보자.
읽기만 하고 아무 파일도 만들지 않는다.
실행: python analysis/02_duplicates_structure.py
"""


# 실행 결과 요약
# 입력이 같은 행(쌍둥이)이 우연이 아님
# 입력이 다른 이웃 행은 보통 21개 특징 열 중 9개가 다른데 쌍둥이는 21개 전부 같은 것으로 출력됨
# Shot 번호가 같은 행은 항상 2행이고 입력도 동일함
# 4행 묶음은 연속된 두 Shot 번호가 2행씩 존재함
# 두 행이 모두 train에 있는 쌍 1,393개 중 97.7%(1,361쌍)는 원본 불량 기록 26열이 완전히 같음
# 독립된 두 부품이라고 하기에는 불량 기록이 같은 쌍의 비율이 너무 높아서 두 행은 같은 샷 기록을 공유하는 것으로 보임
# _1/_2 열은 쌍의 첫째 행/둘째 행과 대응하지 않는 것으로 추정됨
# Shot 번호가 달라도 입력이 같은 경우(256개 묶음, 804행)와 한 행뿐인 Shot 번호(1,699개)가 생기는 이유는 아직 설명할 수 없음

import common
import numpy as np
import pandas as pd

common.announce("02 쌍둥이 행 구조 조사", "몇 초", "몇 초")

ID_COLS = ["shot_id", "id", "Shot"]

# ---------------------------------------------------------------
# 파일 읽기
# ---------------------------------------------------------------
X = pd.read_csv("data/주조공정_입력.csv", header=1, encoding="utf-8-sig")
X.columns = X.columns.str.strip()
labels = pd.read_csv("정답_train.csv", encoding="utf-8-sig")
defects = pd.read_csv("참고_원본불량기록_train.csv", encoding="utf-8-sig")

constant_cols = []
for c in X.columns:
    if X[c].nunique(dropna=True) <= 1:
        constant_cols.append(c)

feature_cols = []
for c in X.columns:
    if c not in ID_COLS + constant_cols:
        feature_cols.append(c)

# [준비] 입력(21개 특징)이 바로 앞 행과 같은지 판정.
# id는 모든 행이 달라서 포함하면 같은 입력을 잡지 못하기 때문에 식별 열 제외하고 비교.
values = X[feature_cols].fillna(-999).values
same_as_prev = np.r_[False, (values[1:] == values[:-1]).all(axis=1)]

# group: 입력이 같은 행들의 묶음 번호 (preprocess.py의 group 열과 같은 방식)
# shot_run: Shot 번호가 같은 연속 행들의 묶음 번호 (입력 기준 묶음과 비교하려고 따로 생성)
X["group"] = np.cumsum(~same_as_prev) - 1
X["shot_run"] = (X["Shot"] != X["Shot"].shift()).cumsum()

# ---------------------------------------------------------------
# 1. 입력이 같은 묶음 vs Shot 번호가 같은 연속 구간
#    목적: '입력이 같다'는 현상과 'Shot 번호가 같다'는 현상이 같은 것인지 판단.
#          같다면 Shot 번호가 쌍둥이를 설명하는 단서가 됨.
# ---------------------------------------------------------------
print("=== 1. 입력이 같은 묶음 vs Shot 값이 같은 연속 구간 ===")
groups = X.groupby("group").agg(
    size=("shot_id", "size"),
    n_shot_values=("Shot", "nunique"),
)
# 표의 세로 = 묶음 크기(묶음에 속한 행 수), 가로 = 그 묶음 안의 Shot 번호 종류 수.
# 칸의 숫자 = 그 두 조건에 맞는 묶음의 개수.
print(
    pd.crosstab(
        groups["size"],
        groups["n_shot_values"],
        rownames=["묶음 크기"],
        colnames=["묶음 안 Shot 값 종류"],
    ).to_string()
)

# 반대로, Shot 번호가 같은 연속 구간 안에 입력이 같은 묶음이 몇 개 있는지.
runs = X.groupby("shot_run").agg(
    size=("shot_id", "size"),
    n_groups=("group", "nunique"),
)
print("Shot이 같은 연속 구간 크기:", runs["size"].value_counts().sort_index().to_dict())
print("그 구간 안의 입력 묶음 수:", runs["n_groups"].value_counts().to_dict())

# ---------------------------------------------------------------
# 2. 우연의 일치인가?
#    목적: 21개 열이 전부 같은 것이 우연일 수 있는지 판단.
#         입력이 다른 이웃 행들이 평균적으로 여러 열이 다르다면, 21개가 전부 같은 건 우연이 아닐 것.
# ---------------------------------------------------------------
print("\n=== 2. 우연의 일치인가: 입력이 다른 이웃 행은 몇 개 열이 다른가 ===")
# 이웃한 두 행마다 값이 다른 열의 개수를 센 뒤, 입력이 서로 다른 이웃 행 쌍(~same)만 남김.
n_diff_cols = (values[1:] != values[:-1]).sum(axis=1)[~same_as_prev[1:]]
print(
    "21개 특징 중 달라진 열 수:",
    f"중앙값 {int(np.median(n_diff_cols))}",
    f"평균 {n_diff_cols.mean():.1f}",
    f"최소 {n_diff_cols.min()}",
)
print(f"1~2개 열만 다른 경우: {int((n_diff_cols <= 2).sum())} / {len(n_diff_cols)}")

# ---------------------------------------------------------------
# 3. Shot 번호 흐름 (묶음 단위)
#    목적: Shot 번호가 규칙적으로 오르는지, 건너뛰는지 보고 묶음의 성격을 짐작.
# ---------------------------------------------------------------
print("\n=== 3. Shot 번호 흐름 ===")
first_shot = X.groupby("group")["Shot"].first()
step = first_shot.diff().dropna()  # 묶음 사이에서 Shot 번호가 얼마나 올랐는가
print("묶음 사이 Shot 증가폭:", step.value_counts().head(8).to_dict())

# 4행 묶음은 Shot 번호가 어떻게 생겼는지 직접 확인 (예: [2, 2, 3, 3]이면 두 번호가 2행씩).
four_row_groups = groups.index[groups["size"] == 4]
examples = []
for k in four_row_groups[:5]:
    examples.append(X.loc[X["group"] == k, "Shot"].tolist())
print("4행 묶음의 Shot 패턴 예:", examples)

n_rising = 0
for k in four_row_groups:
    shot_values = X.loc[X["group"] == k, "Shot"].values
    # 번호가 한 번이라도 오르면(차이가 0보다 크면) 셈.
    if (np.diff(shot_values) > 0).any():
        n_rising += 1
print("Shot이 중간에 오르는 4행 묶음 수:", n_rising, "/", len(four_row_groups))

# ---------------------------------------------------------------
# 4. 원본 불량 기록의 _1 / _2 와 묶음
#    목적: 쌍둥이의 두 행이 '서로 다른 부품'이라면 불량 기록도 따로 적혔을 것이라고 생각됨.
#         기록이 거의 같다면 두 행은 사실상 같은 기록을 공유하는 것으로 추정.
# ---------------------------------------------------------------
print("\n=== 4. 원본불량기록 _1/_2 와 묶음 ===")
slot1_cols = []
slot2_cols = []
for c in defects.columns:
    if c.endswith("_1"):
        slot1_cols.append(c)
    if c.endswith("_2"):
        slot2_cols.append(c)
defects["n1"] = defects[slot1_cols].sum(axis=1)  # 행마다 _1 열에 찍힌 불량 합
defects["n2"] = defects[slot2_cols].sum(axis=1)  # 행마다 _2 열에 찍힌 불량 합

only1 = int(((defects["n1"] > 0) & (defects["n2"] == 0)).sum())
only2 = int(((defects["n1"] == 0) & (defects["n2"] > 0)).sum())
both = int(((defects["n1"] > 0) & (defects["n2"] > 0)).sum())
none = int(((defects["n1"] == 0) & (defects["n2"] == 0)).sum())
print(
    f"한 행에서 _1열에만 / _2열에만 / 두 열 모두 / 어느 열에도 없음: {only1} / {only2} / {both} / {none}"
)

# train 행만 (불량 기록이 train에만 있다) 사용해서, 묶음 안에서 몇 번째 행인지(첫째, 둘째...)별로 불량률을 구함.
# 목적: _1 열이 '묶음의 첫째 행', _2 열이 '둘째 행'을 뜻한다면 행 순서별 불량률이 크게 달라야 함.
m = X.merge(defects, on="shot_id").merge(labels, on="shot_id")
m["position"] = m.groupby("group").cumcount()  # 묶음 안에서 몇 번째 행인지
m["group_size"] = m.groupby("group")["shot_id"].transform("size")


def share_positive(values):
    """값이 0보다 큰(불량이 찍힌) 행의 비율."""
    return (values > 0).mean()


for size in [1, 2, 4]:
    sub = m[m["group_size"] == size]
    rate1 = sub.groupby("position")["n1"].apply(share_positive).round(3).to_dict()
    rate2 = sub.groupby("position")["n2"].apply(share_positive).round(3).to_dict()
    print(
        f"  묶음 크기 {size} (train 행 {len(sub)}): 위치별 _1 불량률 {rate1} | _2 불량률 {rate2}"
    )

# ---------------------------------------------------------------
# 5. 쌍(2행)의 두 행은 불량 기록도 같은가?
#    목적: 입력이 같은 두 행의 불량 기록(26열)이 완전히 같은 비율을 셈.
#         독립된 두 부품이라면 훨씬 자주 달라야 함.
# ---------------------------------------------------------------
print("\n=== 5. 쌍(2행)에서 두 행의 불량 기록 ===")
slot_cols = slot1_cols + slot2_cols  # 불량 기록 26열

n_same = 0
n_diff = 0
different_pairs = []  # 기록이 다른 쌍의 상세 (아래에서 예시 출력과 _1/_2 열 분석에 사용함)
for _, pair in m.groupby("group"):
    if len(pair) != 2:  # 두 행이 모두 train에 있는 쌍만 본다
        continue
    first, second = pair.iloc[0], pair.iloc[1]
    if (first[slot_cols].values == second[slot_cols].values).all():
        n_same += 1
    else:
        n_diff += 1

        # 각 행에서 불량이 찍힌 열의 이름을 모음.
        first_defect_cols = []
        second_defect_cols = []
        for c in slot_cols:
            if first[c]:
                first_defect_cols.append(c)
            if second[c]:
                second_defect_cols.append(c)

        different_pairs.append(
            (
                int(first["shot_id"]),
                int(second["shot_id"]),
                first["label"],
                second["label"],
                first_defect_cols,  # 첫 행에 찍힌 불량 열
                second_defect_cols,  # 둘째 행에 찍힌 불량 열
            )
        )

print(
    f"train에서 두 행이 모두 있는 쌍: {n_same + n_diff}",
    f"| 불량 기록 26열까지 완전히 같은 쌍: {n_same}",
    f"| 다른 쌍: {n_diff}",
)

print("다른 쌍 예시 8개 (shot_id 2개, 라벨 2개, 불량 열 2개):")
for row in different_pairs[:8]:
    print("  ", row)

first_normal = 0
for pair_info in different_pairs:
    if pair_info[2] == "정상":  # 첫 행의 라벨이 정상인 쌍
        first_normal += 1
print(
    f"다른 쌍 중 첫 행이 정상인 경우 / 아닌 경우: {first_normal} / {len(different_pairs) - first_normal}"
)

# 기록이 다른 쌍에서, 불량이 _1 열에만 찍혔는지 _2 열에만 찍혔는지 셈.
slot_use = {"_1열에만": 0, "_2열에만": 0, "두 열 모두": 0}
for row in different_pairs:
    for cols in (row[4], row[5]):
        if not cols:  # 불량이 없는 행은 건너뜀
            continue
        # 이 행의 불량이 _1 열에 찍혔는지, _2 열에 찍혔는지 확인.
        has1 = False
        has2 = False
        for c in cols:
            if c.endswith("_1"):
                has1 = True
            if c.endswith("_2"):
                has2 = True

        if has1 and has2:
            key = "두 열 모두"
        elif has1:
            key = "_1열에만"
        else:
            key = "_2열에만"
        slot_use[key] += 1
print("다른 쌍의 불량이 찍힌 위치(불량이 있는 행 기준):", slot_use)
